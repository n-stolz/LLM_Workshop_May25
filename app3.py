import streamlit as st
import os
import json
import io

import pandas as pd
import openai
import anthropic
import fitz  
import tiktoken
import json
import matplotlib.pyplot as plt

from PIL import Image
import base64




import pymupdf4llm

# --- Session state for login ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# --- Login page ---
def login_page():
    st.title("🔐 Research Assistant Workshop Login")

    password_input = st.text_input("Enter workshop password", type="password")
    if st.button("Login"):
        if password_input == st.secrets["general"]["password"]:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")

# --- Post-login setup ---
def load_resources():
    st.session_state["openai_key"] = st.secrets["api_keys"]["openai_api_key"]
    st.session_state["anthropic_key"] = st.secrets["api_keys"]["anthropic_key"]

    openai.api_key = st.session_state["openai_key"]
    anthropic.api_key = st.session_state["anthropic_key"]

    with st.spinner("🔍 Loading reports"):
     # Read the pre-loaded files (already converted to markdown so save some time during workshop - ~30sec-1min per pdf)
        with open("reports/VCS2410.md", "r", encoding="utf-8") as f:
            VCS2410 = f.read()

        with open("reports/VCS2834.md", "r", encoding="utf-8") as f:
            VCS2834 = f.read()

        with open("reports/GCSP1025.md", "r", encoding="utf-8") as f:
            GCSP1025 = f.read()
        st.session_state["project_dict"] = {
            'VCS2410': VCS2410, 
            'VCS2834': VCS2834,
            'GCSP1025': GCSP1025
        }
        st.session_state["user_prompt"] = """
        **Example User Prompt**:
        user_prompt =  
        <objective>
          Extract structured information about project actors involved in carbon projects based on the unstructured project descriptions. Follow the guidelines strictly and ensure outputs adhere to the provided schema.
        </objective>
                    
        <instructions>
        You are tasked with extracting structured information about project actors involved in carbon projects based on the project descriptions. Follow the guidelines strictly and ensure outputs adhere to the provided schema.
        Only use information explicitly stated in the project description. If information is not clearly linked to the project or entities, write 'Not found'. If entities are mentioned but are not involved in the project, skip them.

        Project Actors are entities or organizations involved in the project. They can be for-profit companies, non-profit organizations, governments, research institutions, NGOs, cooperatives, etc. Each actor can have multiple roles in the project. They need to have an active role and are not only listed as a stakeholder in the project.

        Company names in the prompt examples are anonymized to Company A, B, C, D... so you understand the context, but the information from the task description does not leak into the output.
        </instructions>
    
        This user prompt would help the model to correctly extract and organize information about the actors involved in the carbon projects by following the predefined guidelines and structured prompt schema.
        """
        st.session_state['output_structure'] = """
    Please strictly follow this JSON output structure - don't add any other information or text to your output:

    {
    "actors": [
            {
                "actor_name": "example actor",
                "activities": ["Activity 1", "Activity 2"],
                "role": ["role A", "role B"],
                "headquarter": "Location",
            }
        ]
    }
    """

        st.session_state["category_description"] = """
  <categories>

  <category1>
  actor_name: The name of the actor is the legal name of the entity. Stakeholders only mentioned as consulted stakeholders in Stakeholder Consultation (e.g. "village of Rishabbdeo") are no actors, since they have no active role in the project. However, organisations conducting stakeholder consultations are actors. 
  Actors must be concrete organisations or a organised group of people (company, cooperative, NGO,...) and not a unorganised group of people (Field Agents/Production Managers/Farmers/Local NGOs).
  <examples>
    -Company prepares PDD (indicated by prepared by...)
    -Company is resoponsible of Operaitons
    -Company is responsible for selling carbon credits
    -Company is responsible for technology provision
    -Company is responsible for project development
    -Company is responsible for project implementation
    -Company is responsible for project management
    -Company has any other active role in the project


  <category2>
  role: Each actor has at least one role in the project. We distinguish between the following roles in the carbon credit value chain. Make absolutely sure to only use role names listed here and do not use any other names:
    - "Primary Sales"
    - "Carbon Services"
    - "Technology Provision"
    - "Operation"
    - "Registry & Standard"
    - "Validation & Verification"
    - "Land Steward/land owner"
    - "Purchaser/Funder"
    - "Research partners"
    - "Other" 
    The roles are defined as:

  <role1>
  1. Primary Sales:
        The entity that holds the original ownership of the carbon credits generated by a project.
        Primary Sales does not refer to the sale of other products or benefits from the project (e.g., timber, crops). It includes:
            - The project proponent or project owner.
            - The landowner or entity responsible for the project's operations if they hold the carbon rights.
            - An entity explicitly responsible for selling the carbon credits, even if carbon rights are not formally detailed in the document.

        Key Indicators: 
            - The entity is contractually assigned the carbon rights.
            - The entity is described as the legal owner of the project's Verified Emission Reductions (e.g., CERs, VERs).

        Notes:
            -Focus on carbon credit ownership and sales only, not broader project product sales.
      <examples>
      <example1>
          -"All the revenue from the usufructs of trees planted will belong to the communities. Only the carbon rights will belong to the VPA Implementer/CME, from which revenues will be generated to implement the project. [....] company A as the CME and company B
           as the Project Implementer are involved in the project activity. [...] The GS VERs from the trees are with CME, OffsetFarm Pte. Ltd. This is transferred through end user agreements to the CME/VPA Implementer." -> actor_name="company A", role="Primary Sales"
      </example1>
      <example2>
          -"The Project Proponent is company a, as stated in the PD, represented by Carter Coleman and Theron Morgan-Brown. [...] The Lease/Carbon Rights contracts, signed by the District, Village Council and Individual landowners give copmany a the right to produce and sell carbon credits, and the CCROs assure the long-term security of the owner." -> actor_name="company A", role="Primary Sales"
      </example2>
      <example3>
          -"Subsequently, company A has a contractual agreement with the lead project proponent company B transferring the carbon rights of the project to company B. This project development and project ownership agreement between company A and company B clearly states, that the lead project developer company b is the legal owner of the project and of its Verified Emission Reductions “VERs” under this cooperation throughout the agreement," -> actor_name="company B", role="Primary Sales"
      </example3>
  </role1>

  <role2>
  2. Carbon Services: 
    Services provided by entities to support the development and management of carbon credit projects. Activites include:

    1. Pre-Project Assistance:
        -Project identification
        - Preparation of Project Design Documents (PDD)
        - Feasibility studies related to carbon potential
        - Environmental Impact Assessments

    2. Methodology Know-How:    
        - Development of carbon methodologies
        - Baseline scenario development

    3. Registration & Monitoring Design:
        - Due diligence for project registration
        - MRV (Monitoring, Reporting, and Verification) system design
        - Assigning registration and issuance tasks
        - Aggregating monitoring data to meet standard requirements
        - Developing monitoring plans
        - Coordination with carbon registries and standards bodies

    4. Commercial Know-How related to carbon credits:
        - Commercial advisory services
        - Structuring of Emission Reduction Purchase Agreements (ERPAs)

    5. Consultance on carbon credit project development.
        - Consulting services on any aspect of carbon credit project development if related to the above activities (e.g. technical advisor, strategic consultant...)


    Notes:
        - Field data collection is classified under Operation, not Carbon Services. However, aggregation of field data for preparing reports for the standard, registry or validator is a Carbon Service.
        - Verification and validation of project documents are not considered Carbon Services.
        - Technical support, advising, consulting, or assistance is only counted as Carbon Service if it directly relates to activities around carbon credit development (as listed above), not if relates to Operation acitivities.
        - It is not Carbon Service if it relates to on-the-ground project activities, training of participants, or project operations.

        Preparing a Carbon Project Description document is considered a Carbon Service activity. The company or entity that prepared the description is classified as a Carbon Service provider.

      <examples>
      <example1>
          -"specialized services on climate change mitigation. Key services offered by company A include consulting to governments, non governmental organizations, and private comp anies in several areas related to the environment and climate change. Company A is involved with the design and development of climate
          change mitigation projects and undertakes related services such as performing project baseline studies, designing and implementing monitoring plans and identification of project developers and sources of funding for projects." -> actor_name="Copmany A", role="Carbon Services"
      </example1>
      <example2>
          - "Organization name Company A Role in the project Project Proponent; VCS technical advisor and project partner responsible for VCS project development" -> actor_name="Company A", role="Carbon Services"
          The Hongera Reforestation Project (Mt Kenya and Aberdares) is a reforestation and afforestation initiative designed and funded by Company A , with technical support from Company B, and implemented by Company C. -> actor_name="Company B", role="Carbon Services", Explanation: Technical support in project descriptions usually referes to carbon services
      <example3>
          - "Document Prepared by company A " -> actor_name="company A", role="Carbon Services". Explanation: Preparing the project description document is a carbon servce
      </example3>
      <example4>
          - "company A is the CME of the project" -> actor_name="Company A", role="Carbon Services". Explanation: The Coordinating/Managing Entity (CME) is responsible for project development and carbon services under Gold Standard
      </exmaple5>
      </examples>
  </role2>

  <role3>
  3. Technology Provision: 
       Supplying or developing technology that supports project activities. Includes activities such as:
        - Developing or providing Original Equipment Manufacturer (OEM) technologies (e.g. biochar kiln, monitoring equipment, other machinery)
        - Creating or supplying digital Monitoring, Reporting, and Verification (dMRV) applications or systems
        - Providing technologies designed to incentivize participant engagement
        - Other technological solutions relavant to the project development or operation

        Notes:
        - Technology Provision is recognized when the project actively uses the technology provided or developed by the actor.
        - Merely using or maintaining a dMRV system does not qualify as Technology Provision if the system was developed by another organization.

      Provide technology for projects (e.g. OEMs, developers of dMRV app/technology, technologies to incentivise participation). We consider it as activ project participation if the project uses the technology provided or developed by an actor. Usage or maintanance dMRV systems in project is not technology provision if the system is developed by other organisation.
      <examples>
      <example1>
          -"Internal documentation and logged into the VarunaCarbon dMRV system developed by company A" -> actor_name="company A", role="Technology Provision"
      </example1>
      <example2>
          -"Benefits include a livestock-to-market mechanism provided by our commercial partner companz A. The MNP organizes mobile auctions that bring rural farmers and commercial buyers together; provides livestock management training for herders, NGOs, and farmers; and organizes mobile abattoirs, enabling increased market opportunities for farmers and providing NGOs and farming communities with bulk purchasing power and access to critical farming equipment and vaccinations." -> actor_name="Company A", role="Technology Provision"
      </example2>
      </examples>
  </role3>

  <role4>
  4. Operation: 
        Involves managing, conducting, or coordinating activities directly on the ground related to carbon project execution. Operation is also often referred to as Implementation. Includes tasks such as:
            - Local contracting
            - Resource management
            - Hiring and managing local personnel
            - Communication and coordination with project participants and local stakeholders
            - Conducting participant training related to project activities
            - On-the-ground data collection for monitoring purposes
            - Organizing, supervising and executing project activities (e.g., logging, planting)
            - Operating nurseries for seedling production
            - Agricultural or timber value chain activities that are part of the project or directly linked to the project (e.g., training local community members, land management, agricultural/forestry practices)

        Notes:
            - Field data collection for Monitorin, Reporting and Verification (MRV) is Operation. However, setting up the MRV plan is a Carbon Service. Companies that provide a digital MRV system are considered Technology Provision.
            - Technical support activities are considered Operation if they involve implementation or project execution (e.g., training local community members, land management, agricultural/forestry practices). If they are related to carbon services (e.g., methodology development, baseline development, MRV design, Project Description Design), they are considered Carbon Services.
            - Operation is understood as an ongoing process. Single studies or reports for example on local communities or stakeholders that are comissioned by the project are not considered Operation, but Research Partners.
           


      <examples>
      <example1>
          -"The Hongera Reforestation Project (Mt Kenya and Aberdares) is a reforestation and afforestation initiative designed and funded by company A, with technical support from company B , and implemented by copmany C." -> actor_name="copmany C", role="Operation" . Explanation: Implementation of projects is an operational role
      </example1>
      <example2>
          -"company A, and raised to viable seedling or sapling planting age (which is species-dependent) in nurseries established by company A or the project." -> actor_name="Company A", role="Operation". Explanation: Planting trees is an operational role
      </example2>
      <example3>
          -"These activities will be conducted by local company A members, who are trained by company B technical specialists, supported by the copmany C team and Copmany D advisors." -> actor_name="company B", role="Operation". Explanation: Training of local community members is an operational role
      </example3>
      <example4>
          -"company A is the VPA implementer." -> actor_name="Company A", role="Operation". Explanation: Voluntary program of activity (VPA) implementer is the local implementer of Gold Standard projects.
      </examples>
  </role4>

  <role5>
  5. Registry & Standard: 
    The organization or system under which a carbon project is registered and certified.

    Common Standards:
        - Verified Carbon Standard (VCS)
        - Gold Standard
        - Carbon Standards International
        - Plan Vivo 
        - Puro
        - Isometric

    Notes:
        - If a standard or registry other than those listed above is referenced, confirm carefully that it is indeed acting as a Registry or Standard before assigning this role.
        - Focus only on entities responsible for certifying, issuing, or overseeing carbon credit registration and verification.
      <examples>
      <example1>
          -"Carbon Registry: registry A" -> actor_name="registry A", role="Registry & Standard"
      </example1>
      </examples>
  </role5>

  <role6>
  6. Validation & Verification: 
      -Validation & Verification Bodies. These are always third party organisations that are not involved in the project otherwise 
      <examples>
      <example1>
          -"company A has validated that the project Udzungwa Corridor Reforestation is in compliance with the Verified Carbon Standard version 4 and the CCB Standards" -> actor_name="copmany A", role="Validation & Verification". Reason: AENOR is a validated project against standard
      </example1>
      <example2>
         -"Validation Body Company A 2000 Powell Street, Ste. 600 Emeryville CA 94608 USA www.SCSglobalServices.com" -> actor_name="Company A", role="Validation & Verification"
      </example2>
      </examples>
  </role6>


  <role7>
  7. Land Steward/land owner: 
    - The entity that owns, manages, or holds responsibility for the land on which the project is implemented.
    - This can include ownership with or without a direct economic interest in the land.
   

    Includes:
        - Ministries or government agencies
        - Conservation NGOs
        - Private companies
        - Any entity holding land titles or responsible for distributing land titles

    Notes: 
        - Entities that own the land are always assigned the Land Steward/Land Owner role, regardless of their level of economic involvement.
        - Stewardship can involve either direct land ownership or administrative control over land title distribution.
        - Being the entity that leases land from a landonwer does not qualify as landonwer or land steward
      <example1>
          -"Company A Region Non-Profit Company (K2C) is the implementing agency managing the 2,474,700 ha (UNESCO) Kruger to Canyons biosphere reserve since 2001. K2C is currently running 12 projects with partners in the biosphere linking sustainable development and biodiversity conservation including expansion of protected areas and sustainable land management" -> actor_name="Company A", role="Land Steward"
      </example1>
      </examples>
  </role7>

  <role8>
  8. Purchaser/Funder: 
    Entities that commit funding or financial support to carbon projects. It includes:
       - Grant funding
       - Loans
       - Long-term offtake agreements (e.g., future purchase of carbon credits)
       - Direct investment into the project
       - Initial financing of project equipment
       - Internal funding or financing of equipment by project participants

    Notes:
       - If funding is provided through a subsidiary, the parent company is recorded as the Purchaser/Funder.
       - Focus on entities providing financial resources that enable project development or operation, not just buyers of final carbon credits unless it is through long-term financial commitments.

    
      <examples>
      <example1>
          -"company A’s audit team accessed the three VERPAs (Verified Emission Reductions Purchase Agreements) signed with the following off-takers:
              - company B .
              - company C.
              - company D ." -> actor_name="Company B", role="Purchaser/Funder"; actor_name="Company C", role="Purchaser/Funder; "Company D", role="Purchaser/Funder
      </example1>
      <example2>
          -"company A is a co-funder to CSA for the Pro-Nature Enterprises Project" -> actor_name="Company A", role="Purchaser/Funder"
      </example2>
        <example3>
            -"Company A is an investor in the project" -> actor_name="Company E", role="Purchaser/Funder"
        </example3>
        <example4>
            -"The project equipment is financed by company F" -> actor_name="Company F", role="Purchaser/Funder"
        </example4>
      </examples>
  </role8>

  <role9>
  9. Research partners: 
      Research institutions, universities, or NGOs that conduct research or provide studies directly for the project that are not primarily related to carbon credit development or carbon credit methodologies.
      Includes:
        - Conducting feasibility studies or environmental/social impact studies that are important for the project’s development but not tied to carbon credit methodologies.
        - Providing baseline studies that focus on non-carbon-related aspects (e.g., biodiversity studies, socio-economic assessments).
        - General scientific or technical research aimed at understanding or supporting the project context, but not specific to carbon measurement or crediting.
        - Social assessments, community impact studies, or other research that informs project design or implementation but does not focus on carbon credit methodologies.

      Excludes:
        - Carbon-related studies such as modeling carbon stocks, carbon sequestration estimation, or developing carbon methodologies, which are part of Carbon Services.
        - If the research is primarily focused on developing or supporting carbon credit methodologies (e.g., GHG emissions modeling), this falls under Carbon Services, not Research Partners.
        
      Notes:
      - Research Partners are only considered actors if they conduct studies specifically commissioned for the project.
      - If an organization is cited for their published research in a general sense but is not directly involved in the project, they are not considered a Research Partner.

      <examples>
      <example1>       
          "The project proponent issued as study with university A on the social implications of the project" -> actor_name="university A", role="Research partners"
      </example1>
       
      </examples>
  </role9>

  <role10>
  10. Other:
   Definition:
    This role is for entities whose involvement in the project does not clearly fit into any of the previously defined roles. It serves as a catch-all category for undefined or unclear roles that do not directly relate to project implementation or carbon services. Onlyuse this category if the actor does not fit any other category. It includes:
     - Entities providing general consulting or approval for the project (e.g., a government agency providing oversight but not actively engaged in project implementation).
     - Entities involved in community engagement activities not explicitly tied to training for project implementation (if not directly involved in training, this would fall under Operation).
     - Entities providing legal advice
    
    Excludes:
    - If an entity is already categorized under one of the other roles (e.g., Land Steward, Carbon Services, Operation), do not add "Other" as an additional role for the same entity.
    - Roles that are directly related to carbon credit generation or project operations (those should be categorized appropriately under Carbon Services, Operation, etc.).

    Notes:
    - It serves as a miscellaneous category, but should not be overused when an entity clearly fits within another role.
    
    <examples>
      <example1>       
          "Government agency A providing consultation or approval for the project without having a defined role in implementation." -> actor_name="Government Agency A A", role="Other"
      </example1>
       
      </examples>


  </role10>
  </category2>

  <category3>
  activities: 
      Concrete actions that the actor is taking in the project. Activities must be named for an oragnisation in the project description. Concrete acitivites relate to project roles.
      <examples>
      <example1>
          -"These two villages are the ones that have already reached agreements with company A, whereas Idunda and Kidete have not yet joined the carbon project (this is a grouped project)." -> actor_name="company A", activities=["Reached agreements with villages"]
      </example1>
      <example2>
          -"The company A is responsible for the planting of trees, thinning, and pruning" -> actor_name="Copmany A", activities=["Planting trees", "Thinning", "Pruning"]
      </example2>
       <example3>
            "NGO B engaged with local communities to work out a benefit sharing plan" -> actor_name="universityz A", role="Research partners"
        </example3>
      </examples>
  </category3>

  <category4>
  headquarter: 
      Headquarter of the actor. This can be a city, country or region. Be as granular as possible while sticking to given information.
      <examples>
      <example1>
          -"As stated in the PD, Udzungwa Corridor Limited is a registered company in Tanzania, No. 151884695 /259/, with a Board of Directors." -> headquarter="Tanzania"
      </example1>
      <example2>
          -"Organization name
              Company A
              Role in the project
              Property Manager
              Contact person
              Alvaro Perez del Castillo
              Title
              General Manager
              Address
              Costa Rica 1566 , Montevideo
              Telephone
              +598 95630638
              Email
              alvaro.perez@pike.com.uy" -> headquarter="Montevideo, Uruguay"
      </example2>
      </examples>
  </category4>
  
  </categories>
  """

# --- Main multi-task app ---
def main_app():
    st.sidebar.title("🧪 Workshop Tasks")
    task = st.sidebar.radio("Select Task", [f"Task {i}" for i in range(1, 7)])

    if "project_dict" not in st.session_state:
        load_resources()

    if task == "Task 1":
        task_1()
    elif task == "Task 2":
        task_2()
    elif task == "Task 3":
        task_3()
    elif task == "Task 4":
        task_4()
    elif task == "Task 5":
        task_5()
    elif task == "Task 6":
        task_6()

# --- Placeholder task pages ---
import matplotlib.pyplot as plt
import tiktoken
import streamlit as st

def task_1():
    st.header("📊 Task 1: Estimating API Request Costs of Reading Different Carbon Project Descriptions")

    st.markdown("""
    Using the API of Large Language Models always comes with a cost. However, different models are priced differently.  
    To use an LLM, one needs to create an account, generate an API key, and add credit to the account.  
    A tutorial for GPT is [here](https://www.merge.dev/blog/chatgpt-api-key), and one for Anthropic is [here](https://www.merge.dev/blog/anthropic-api-key).

    **Never** upload your key anywhere (e.g. GitHub, paper submission...). If it gets exposed, people can use credit from your account!

    **Length of Documents (PDFs include images, tables, etc.):**
    - **VCS2410**: 141 pages  
    - **VCS2834**: 236 pages  
    - **GCSP1025**: 50 pages
    """)

    # Add download buttons for the PDFs
    st.subheader("Download Project Reports")

    st.markdown("""
                Here you can download the project descriptions and look at them. We will use them during the whole workshop.
                """)
    st.download_button(
        label="Download VCS2410 PDF",
        data=open("reports/VCS2410.pdf", "rb").read(),
        file_name="VCS2410.pdf",
        mime="application/pdf"
    )
    st.download_button(
        label="Download VCS2834 PDF",
        data=open("reports/VCS2834.pdf", "rb").read(),
        file_name="VCS2834.pdf",
        mime="application/pdf"
    )
    st.download_button(
        label="Download GCSP1025 PDF",
        data=open("reports/GCSP1025.pdf", "rb").read(),
        file_name="GCSP1025.pdf",
        mime="application/pdf"
    )

    # Pricing Table (May 2025)
    pricing = { 
        "gpt o3": {"input": 10.0, "output": 40.0},
        "gpt 4.1": {"input": 2.0, "output": 8.0},
        "gpt 4.1-mini": {"input": 0.4, "output": 1.6},
        "claude-3-opus": {"input": 15.0, "output": 75.0},
        "claude-3-sonnet": {"input": 3.0, "output": 15.0},
    }

    expected_output_tokens = st.slider("Expected output tokens", 100, 10000, 800, step=100)

    encoding = tiktoken.get_encoding("cl100k_base")
    project_dict = st.session_state["project_dict"]

    selected_doc = st.selectbox("Choose a document", list(project_dict.keys()))

    input_text = project_dict[selected_doc]
    input_tokens = encoding.encode(input_text)
    num_input_tokens = len(input_tokens)

    # Calculate costs
    cost_data = {}
    for model, price in pricing.items():
        cost_input = (num_input_tokens / 1_000_000) * price["input"]
        cost_output = (expected_output_tokens / 1_000_000) * price["output"]
        total_cost = cost_input + cost_output
        cost_data[model] = total_cost

    # Plotting
    models_list = list(cost_data.keys())
    cost_values = list(cost_data.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models_list, cost_values, color='skyblue')
    ax.set_title(f"Estimated Cost per Model for {selected_doc}")
    ax.set_ylabel("Cost (USD)")
    ax.set_xlabel("Model")
    ax.set_ylim(0, max(cost_values) * 1.3)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for bar, cost in zip(bars, cost_values):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.0002, f"${cost:.2f}",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    st.pyplot(fig)


# Streamlit app for Task 2 - Analyzing PDFs using LLM API
import openai
import streamlit as st

# Streamlit app for Task 2 - Analyzing PDFs using LLM API
def task_2():
    st.header("📝 Task 2: Using LLM API to Analyze PDFs")

    # Description and instructions
    st.markdown("""
    In this task, you'll experiment with system prompts and temperature settings to see how the LLM's response changes.
    
    ### Exercises:
    1. **System Prompt**: Play with different system prompts. How does it change the output?
    2. **Temperature**: Adjust the temperature (range 0-1). What do you observe in the responses?
    """)

    # System prompt options with a choice of predefined prompts or custom prompt
    st.subheader("Choose or Customize a System Prompt")
    
    system_prompt_options = [
        "You are a helpful assistant. Answer the question directly.",
        "You are a poet. Don't answer the question directly, but answer in vague poems.",
        "You are an extremely accurate researcher. Answer the question and cover all caveats.",
        "You are a Swiss German translator. Translate all inputs to Swiss German without adding content."
    ]
    
    # Dropdown to select a system prompt
    selected_prompt = st.selectbox(
        "Choose a system prompt:",
        system_prompt_options
    )
    
    # Option for the user to write a custom system prompt
    custom_prompt = st.text_area(
        "Or, write your own system prompt (leave blank to use selected prompt):",
        value=""
    )

    # Determine the final system prompt based on the user's input
    if custom_prompt:
        system_prompt = custom_prompt
    else:
        system_prompt = selected_prompt

    # Temperature slider
    st.subheader("Temperature Setting")
    temperature_value = st.slider(
        "Choose temperature (0 = less creative, 1 = more creative):",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )

    # User prompt input
    st.subheader("User Prompt (ask anything)")
    user_prompt = st.text_input(
        "Enter a question or prompt for the model",
        value="What color is the sky?"
    )

    # Button to generate response
    if st.button("Generate Response"):
        if user_prompt:
            # Display spinner to show the user that the model is processing the request
            with st.spinner('Generating response... Please wait.'):
                # Get the answer from the LLM API
                response = get_model_response2(system_prompt, user_prompt, temperature_value)
            
            st.subheader("Model Response")
            st.write(response)
        else:
            st.error("Please enter a user prompt.")

def get_model_response2(system_prompt, user_prompt, temperature_value):
    # Set OpenAI API key
    openai_key = st.secrets["api_keys"]["openai_api_key"]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=openai_key)

    MODEL = "gpt-4"
    # Make API request to get response
    response = client.beta.chat.completions.parse(
                        model=MODEL,
                        messages=messages,
                        temperature=temperature_value,
                        max_tokens=1000,
                    )
    output = response.choices[0].message.content
    return output



import openai
import streamlit as st

# Streamlit app for Task 3 - Structured Information Extraction from PDFs
def task_3():
    st.header("📝 Task 3: Get Structured Information from PDFs without Structured Output")

    # Instructions
    st.markdown("""
    The goal of this task is to extract structured information from unstructured text, specifically the names of companies involved in carbon projects.

    ### Exercise:
    1. Write a user prompt that tells the model what to do (e.g., extracting company names).
    2. Review the category descriptions to get a feeling of how to write a codebook for the task.

    The task includes category descriptions that define the role of each actor in the carbon credit value chain. Use these descriptions to guide your prompt formulation.
    """)

    # Collapsible category description
    with st.expander("Click to see Category Descriptions"):
        st.markdown("""
        # Category Description

## Category 1: Actor Name
The name of the actor is the legal name of the entity. Stakeholders only mentioned as consulted stakeholders in Stakeholder Consultation (e.g., "village of Rishabbdeo") are not actors, since they have no active role in the project. However, organisations conducting stakeholder consultations are actors. Actors must be concrete organisations or an organised group of people (company, cooperative, NGO, etc.), not an unorganised group of people (Field Agents/Production Managers/Farmers/Local NGOs).

### Examples:
- Company prepares PDD (indicated by prepared by...)
- Company is responsible for operations
- Company is responsible for selling carbon credits
- Company is responsible for technology provision
- Company is responsible for project development
- Company is responsible for project implementation
- Company is responsible for project management
- Company has any other active role in the project

## Category 2: Role
Each actor has at least one role in the project. We distinguish between the following roles in the carbon credit value chain. Make absolutely sure to only use the role names listed here and do not use any other names:

- "Primary Sales"
- "Carbon Services"
- "Technology Provision"
- "Operation"
- "Registry & Standard"
- "Validation & Verification"
- "Land Steward/Land Owner"
- "Purchaser/Funder"
- "Research Partners"
- "Other"

### Role Definitions:

#### 1. Primary Sales
The entity that holds the original ownership of the carbon credits generated by a project. This does not refer to the sale of other products or benefits from the project (e.g., timber, crops). It includes:
- The project proponent or project owner
- The landowner or entity responsible for the project's operations if they hold the carbon rights
- An entity explicitly responsible for selling the carbon credits, even if carbon rights are not formally detailed in the document.

**Key Indicators:**
- The entity is contractually assigned the carbon rights.
- The entity is described as the legal owner of the project's Verified Emission Reductions (e.g., CERs, VERs).

**Notes:**
- Focus on carbon credit ownership and sales only, not broader project product sales.

##### Examples:
1. "All the revenue from the usufructs of trees planted will belong to the communities. Only the carbon rights will belong to the VPA Implementer/CME, from which revenues will be generated to implement the project. [....] company A as the CME and company B as the Project Implementer are involved in the project activity. [...] The GS VERs from the trees are with CME, OffsetFarm Pte. Ltd. This is transferred through end-user agreements to the CME/VPA Implementer."  
    - actor_name="company A", role="Primary Sales"
2. "The Project Proponent is company a, as stated in the PD, represented by Carter Coleman and Theron Morgan-Brown. [...] The Lease/Carbon Rights contracts, signed by the District, Village Council, and Individual landowners give company a the right to produce and sell carbon credits, and the CCROs assure the long-term security of the owner."  
    - actor_name="company A", role="Primary Sales"
3. "Subsequently, company A has a contractual agreement with the lead project proponent company B transferring the carbon rights of the project to company B."  
    - actor_name="company B", role="Primary Sales"

#### 2. Carbon Services
Services provided by entities to support the development and management of carbon credit projects. Activities include:

1. **Pre-Project Assistance:**
   - Project identification
   - Preparation of Project Design Documents (PDD)
   - Feasibility studies related to carbon potential
   - Environmental Impact Assessments

2. **Methodology Know-How:**
   - Development of carbon methodologies
   - Baseline scenario development

3. **Registration & Monitoring Design:**
   - Due diligence for project registration
   - MRV (Monitoring, Reporting, and Verification) system design
   - Assigning registration and issuance tasks
   - Aggregating monitoring data to meet standard requirements
   - Developing monitoring plans
   - Coordination with carbon registries and standards bodies

4. **Commercial Know-How related to carbon credits:**
   - Commercial advisory services
   - Structuring of Emission Reduction Purchase Agreements (ERPAs)

5. **Consulting on carbon credit project development:**
   - Consulting services on any aspect of carbon credit project development if related to the above activities (e.g., technical advisor, strategic consultant...)

**Notes:**
- Field data collection is classified under Operation, not Carbon Services. However, aggregation of field data for preparing reports for the standard, registry, or validator is a Carbon Service.
- Verification and validation of project documents are not considered Carbon Services.
- It is not Carbon Service if it relates to on-the-ground project activities, training of participants, or project operations.
- Preparing a Carbon Project Description document is considered a Carbon Service activity.

##### Examples:
1. "Specialized services on climate change mitigation, such as performing project baseline studies, designing and implementing monitoring plans."  
    - actor_name="Company A", role="Carbon Services"
2. "Company A is responsible for VCS technical advisory and project development."  
    - actor_name="Company A", role="Carbon Services"
3. "Document prepared by company A."  
    - actor_name="company A", role="Carbon Services"

#### 3. Technology Provision
Supplying or developing technology that supports project activities. Includes:
- Developing or providing Original Equipment Manufacturer (OEM) technologies (e.g., biochar kiln, monitoring equipment, other machinery)
- Creating or supplying digital Monitoring, Reporting, and Verification (dMRV) applications or systems
- Providing technologies designed to incentivize participant engagement
- Other technological solutions relevant to the project development or operation

**Notes:**
- Technology Provision is recognized when the project actively uses the technology provided or developed by the actor.

##### Examples:
1. "Internal documentation logged into the VarunaCarbon dMRV system developed by company A."  
    - actor_name="company A", role="Technology Provision"
2. "A livestock-to-market mechanism provided by company A."  
    - actor_name="Company A", role="Technology Provision"

#### 4. Operation
Involves managing, conducting, or coordinating activities directly on the ground related to carbon project execution. Includes:
- Local contracting
- Resource management
- Hiring and managing local personnel
- Communication and coordination with project participants and local stakeholders
- Conducting participant training related to project activities
- On-the-ground data collection for monitoring purposes
- Organizing, supervising, and executing project activities (e.g., logging, planting)

**Notes:**
- Field data collection for MRV is Operation. However, setting up the MRV plan is a Carbon Service.

##### Examples:
1. "The Hongera Reforestation Project is implemented by company C."  
    - actor_name="Company C", role="Operation"
2. "Company A is responsible for planting trees, thinning, and pruning."  
    - actor_name="Company A", role="Operation"

#### 5. Registry & Standard
The organization or system under which a carbon project is registered and certified.

**Common Standards:**
- Verified Carbon Standard (VCS)
- Gold Standard
- Carbon Standards International
- Plan Vivo
- Puro
- Isometric

##### Examples:
1. "Carbon Registry: registry A"  
    - actor_name="registry A", role="Registry & Standard"

#### 6. Validation & Verification
These are third-party organisations that are not involved in the project otherwise.

##### Examples:
1. "Company A validated the project Udzungwa Corridor Reforestation."  
    - actor_name="Company A", role="Validation & Verification"

#### 7. Land Steward/Land Owner
The entity that owns, manages, or holds responsibility for the land on which the project is implemented.

##### Example:
1. "Company A is the implementing agency managing the Kruger to Canyons biosphere reserve."  
    - actor_name="Company A", role="Land Steward"

#### 8. Purchaser/Funder
Entities that commit funding or financial support to carbon projects, such as grants, loans, or long-term purchase agreements.

##### Examples:
1. "Company A signed VERPAs with companies B, C, and D."  
    - actor_name="Company B", role="Purchaser/Funder"
2. "Company A is a co-funder of the Pro-Nature Enterprises Project."  
    - actor_name="Company A", role="Purchaser/Funder"

#### 9. Research Partners
Research institutions, universities, or NGOs that conduct research or provide studies directly for the project.

##### Example:
1. "University A conducted a study on the social implications of the project."  
    - actor_name="University A", role="Research Partners"

#### 10. Other
This role is for entities whose involvement in the project does not clearly fit into any of the previously defined roles.

##### Example:
1. "Government agency A providing consultation or approval for the project."  
    - actor_name="Government Agency A", role="Other"

## Category 3: Activities
Concrete actions that the actor is taking in the project. Activities must be named for an organisation in the project description.

### Examples:
1. "These two villages have reached agreements with company A."  
    - actor_name="company A", activities=["Reached agreements with villages"]
2. "Company A is responsible for planting trees, thinning, and pruning."  
    - actor_name="Company A", activities=["Planting trees", "Thinning", "Pruning"]

## Category 4: Headquarter
Headquarter of the actor. This can be a city, country, or region.

### Examples:
1. "Company A is a registered company in Tanzania."  
    - headquarter="Tanzania"
2. "Company A is located in Montevideo, Uruguay."  
    - headquarter="Montevideo, Uruguay"

        """)

    # Check if the project descriptions exist in session state
    if "project_dict" not in st.session_state:
        st.error("Project descriptions are not available. Please make sure the session state is initialized.")
        return

    # Let the user choose a project description to work with
    st.subheader("Choose a Project Description")
    project_choice = st.selectbox(
        "Select a project to analyze:",
        options=list(st.session_state["project_dict"].keys())
    )

    # Retrieve the chosen project description
    project_description = st.session_state["project_dict"][project_choice]

    # User prompt input (extraction prompt for companies)
    st.subheader("User Prompt (Tell the model what to do)")
    user_prompt = st.text_area(
        "Write a prompt to instruct the model to extract relevant structured information (e.g., companies involved in the project).",
        value="TODO: write your instruction here."
    )

    # Temperature slider
    st.subheader("Temperature Setting")
    temperature_value = st.slider(
        "Choose temperature (0 = less creative, 1 = more creative):",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )

    # Button to generate response
    if st.button("Generate Response"):
        if user_prompt:
            # Show a spinner to indicate loading
            with st.spinner('Extracting information... Please wait.'):
                # Get the answer from the LLM API
                response = get_model_response(user_prompt, project_description, temperature_value)

            st.subheader("Model Response")
            st.write(response)
        else:
            st.error("Please enter a user prompt.")

    # Expandable solution section
    with st.expander("Possible User Prompt"):
        st.markdown("""
        **Example User Prompt**:
        user_prompt =  
        <objective>
          Extract structured information about project actors involved in carbon projects based on the unstructured project descriptions. Follow the guidelines strictly and ensure outputs adhere to the provided schema.
        </objective>
                    
        <instructions>
        You are tasked with extracting structured information about project actors involved in carbon projects based on the project descriptions. Follow the guidelines strictly and ensure outputs adhere to the provided schema.
        Only use information explicitly stated in the project description. If information is not clearly linked to the project or entities, write 'Not found'. If entities are mentioned but are not involved in the project, skip them.

        Project Actors are entities or organizations involved in the project. They can be for-profit companies, non-profit organizations, governments, research institutions, NGOs, cooperatives, etc. Each actor can have multiple roles in the project. They need to have an active role and are not only listed as a stakeholder in the project.

        Company names in the prompt examples are anonymized to Company A, B, C, D... so you understand the context, but the information from the task description does not leak into the output.
        </instructions>
    
        This user prompt would help the model to correctly extract and organize information about the actors involved in the carbon projects by following the predefined guidelines and structured prompt schema.
        """)

def get_model_response(user_prompt, project_description, temperature_value):
    # Set OpenAI API key
    openai_key = st.secrets["api_keys"]["openai_api_key"]

    # Define the system prompt and category description
    system_prompt = """
    You are a research assistant. Base all your answers on information from carbon project descriptions. Do not add any information that is not in the project description. If you are not sure, write 'Not found'.
    """

    category_description =st.session_state['category_description']
    # Combine system prompt, category description, and user prompt
    full_prompt = user_prompt + "\n" + category_description + f"\nThe carbon project description starts here: \n {project_description[0:100000]}\n"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": full_prompt}
    ]
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=openai_key)

    MODEL = "gpt-4.1-2025-04-14"
    # Make API request to get response
    response = client.beta.chat.completions.parse(
                        model=MODEL,
                        messages=messages,
                        temperature=temperature_value,
                        max_tokens=1000,
                    )
    output = response.choices[0].message.content
    return output



def task_4():
    st.header("📝 Task 4: Adding Structured Output")

    # Instructions
    st.markdown("""
    In this task, we'll provide a clear output structure to the model and parse its response into a structured format.
    
    ### Exercise:
    1. Review the output structure and understand how it will be transformed into a DataFrame.
    2. The model's output will be in JSON format, which we'll parse into a structured table.
    """)

    # Check if the project descriptions exist in session state
    if "project_dict" not in st.session_state:
        st.error("Project descriptions are not available. Please make sure the session state is initialized.")
        return

    # Let the user choose a project description to work with
    st.subheader("Choose a Project Description")
    project_choice = st.selectbox(
        "Select a project to analyze:",
        options=list(st.session_state["project_dict"].keys())
    )

    # Retrieve the chosen project description
    project_description = st.session_state["project_dict"][project_choice]

    # User prompt input
    st.subheader("User Prompt (Tell the model what to do)")
    user_prompt = st.text_area(
    "This is the proposed user prompt to instruct the model to extract relevant structured information (e.g., companies involved in the project).",
    value=(
        "<objective>\n"
        "Extract structured information about project actors involved in carbon projects based on the unstructured project descriptions. "
        "Follow the guidelines strictly and ensure outputs adhere to the provided schema.\n"
        "</objective>\n\n"
        "<instructions>\n"
        "You are tasked with extracting structured information about project actors involved in carbon projects based on the project descriptions. "
        "Follow the guidelines strictly and ensure outputs adhere to the provided schema.\n"
        "Only use information explicitly stated in the project description. If information is not clearly linked to the project or entities, write 'Not found'. "
        "If entities are mentioned but are not involved in the project, skip them.\n\n"
        "Project Actors are entities or organizations involved in the project. They can be for-profit companies, non-profit organizations, governments, "
        "research institutions, NGOs, cooperatives, etc. Each actor can have multiple roles in the project. They need to have an active role and are not "
        "only listed as a stakeholder in the project.\n\n"
        "Company names in the prompt examples are anonymized to Company A, B, C, D... so you understand the context, but the information from the task "
        "description does not leak into the output.\n"
        "</instructions>"
    )
)

    
    # Show the expected output format as guidance
    st.subheader("Expected Output Format")
    st.markdown("We add this output structure to the prompt. The model will fit the desired output structure:")

    st.code("""
    {
    "actors": [
        {
        "actor_name": "example actor",
        "activities": ["Activity 1", "Activity 2"],
        "role": ["role A", "role B"],
        "headquarter": "Location"
        }
    ]
    }
    """, language="json")

    # Temperature slider
    st.subheader("Temperature Setting")
    temperature_value = st.slider(
        "Choose temperature (0 = less creative, 1 = more creative):",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )

    # Define the output structure for the model
    output_structure = st.session_state['output_structure']

    # Combine everything into the full prompt
    full_prompt = user_prompt + st.session_state['category_description'] + f" The carbon project description starts here: \n {project_description[0:100000]}" + output_structure

    # Button to generate response
    if st.button("Generate Response"):
        if user_prompt:
            # Show a spinner to indicate loading
            with st.spinner('Extracting information... Please wait.'):
                # Get the answer from the LLM API
                response = get_model_response4(full_prompt, temperature_value)

            # Extract and clean the JSON output
            output = response
            try:
                # Clean up the response and parse JSON
                cleaned_output = output.strip("```json").strip("```").strip()
                data = json.loads(cleaned_output)

                # Convert JSON data to a pandas DataFrame
                df_actors = pd.json_normalize(data['actors'])

                st.subheader("Model Output - Structured Table")
                st.write(df_actors)
            except Exception as e:
                st.error(f"Error parsing the response: {e}")
        else:
            st.error("Please enter a user prompt.")

def get_model_response4(full_prompt, temperature_value):
    # Set OpenAI API key
    openai_key = st.secrets["api_keys"]["openai_api_key"]

    # Define system prompt
    system_prompt = """
    You are a research assistant. Base all your answers on information from carbon project descriptions. Do not add any information that is not in the project description. If you are not sure, write 'Not found'.
    """



    # Prepare the full prompt for the API
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": full_prompt}
    ]

    # Initialize OpenAI client
    client = openai.OpenAI(api_key=openai_key)

    MODEL = "gpt-4.1-2025-04-14"

    # Make API request to get response
    response = client.beta.chat.completions.parse(
                        model=MODEL,
                        messages=messages,
                        temperature=temperature_value,
                        max_tokens=10000,
                    )
    output = response.choices[0].message.content
    return output

def task_5():
    st.header("📸 Task 5: Extract and Process Images from Project Description")

    st.markdown("""
    As seen in previous tasks, some important information might be hidden in **images** in project descriptions.  
    If we only analyze text, we might **miss key insights**.

    This example shows how to:
    - Extract relevant images
    - Include them in the prompt
    - Use a multimodal model (GPT-4 Vision) to extract structured information

    ### 🔍 How images were selected:
    1. All images were extracted from the PDF (using `PyMuPDF`)
    2. Each image was **classified as relevant or not** using a low-cost model (`gpt-4.1-mini`)
    3. Only relevant images are kept and processed here

    If you’re curious about the classification pipeline, ask me! I'm happy to share more details or code.
    """)

    # Load project descriptions from session
    if "project_dict" not in st.session_state:
        st.error("Session state with project descriptions not found.")
        return

    project_dict = st.session_state["project_dict"]

    # Select a project
    project_id = 'GCSP1025'
    project_description = project_dict[project_id]

    # Load relevant images
    image_folder = f"reports/{project_id}_relevant_images"
    images = load_images_from_folder(image_folder)

    if not images:
        st.warning("No relevant images found for this project.")
        return

    # Display relevant images
    st.subheader("🖼️ Relevant Extracted Images")
    for img in images:
        st.image(base64_to_image(img["image_base64"]), caption=img["filename"], use_container_width=True)

    # User prompt input
    #st.subheader("✍️ Prompt")
    output_structure = st.session_state['output_structure']
    full_prompt = st.session_state["user_prompt"] + st.session_state['category_description'] + f" The carbon project description starts here: \n {project_description}" + output_structure

    # Run model
    if st.button("Run Model"):
        with st.spinner("Running multimodal analysis..."):
            full_output = run_multimodal_gpt(project_id, full_prompt, images, project_dict[project_id])
            if isinstance(full_output, list):
                st.success("Structured data extracted!")
                st.dataframe(pd.DataFrame(full_output))
            else:
                st.error("Failed to parse output. Raw model output:")
                st.text(full_output)


def load_images_from_folder(folder_path):
    image_entries = []
    if not os.path.exists(folder_path):
        return image_entries
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            with open(os.path.join(folder_path, filename), "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode("utf-8")
                image_entries.append({
                    "image_base64": encoded_string,
                    "filename": filename
                })
    return image_entries


def base64_to_image(b64_string):
    return Image.open(io.BytesIO(base64.b64decode(b64_string)))


def run_multimodal_gpt(project_id, user_prompt, images, project_text):
    # OpenAI client
    openai_key = st.secrets["api_keys"]["openai_api_key"]
    client = openai.OpenAI(api_key=openai_key)

    # Build messages
    system_prompt = "You are a research assistant. Extract structured actor data from a carbon project description and its images."
    text_chunk = project_text

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"The following is a carbon project description:\n{text_chunk}\n\n{user_prompt}"}
    ]

    # Append images as separate user content blocks
    for img in images:
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img['image_base64']}"
                    }
                }
            ]
        })

    try:
        response = client.beta.chat.completions.parse(
            model = "gpt-4.1-2025-04-14",
            messages=messages,
            temperature=0,
            max_tokens=8192,
        )

        raw_output = response.choices[0].message.content
        cleaned = raw_output.strip().strip("```json").strip("```").strip()

        # Attempt to parse JSON and extract actors
        data = json.loads(cleaned)
        return data["actors"] if isinstance(data, dict) and "actors" in data else raw_output

    except Exception as e:
        return str(e)


# --- Entry point ---
if __name__ == "__main__":
    if not st.session_state.logged_in:
        login_page()
    else:
        main_app()
