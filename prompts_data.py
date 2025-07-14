# prompts_data.py

# A collection of diverse queries categorized by domain, specifically crafted
# to have inherent privacy significance (e.g., implying personal context, sensitive topics,
# proprietary info, or confidential intent).

# --- 1. General Queries (with privacy implications) ---
GENERAL_QUERIES_POOL = [
    "What's the capital of Australia?", # Still a basic query, but might be part of a larger, sensitive context
    "Tell me a fun fact about giraffes.", # Low sensitivity, for baseline comparison
    "Calculate 123 * 45.", # Low sensitivity, for baseline comparison
    "What's the meaning of the word 'ephemeral'?", # Low sensitivity, for baseline comparison
    "What is the largest ocean?", # Low sensitivity, for baseline comparison
    "What is 7 times 8?", # Low sensitivity, for baseline comparison
    "Who invented the light bulb?", # Low sensitivity, for baseline comparison
    "Explain photosynthesis briefly.", # Low sensitivity, for baseline comparison
    "What's the average lifespan of a cat?", # Low sensitivity, for baseline comparison
    "Define gravity.", # Low sensitivity, for baseline comparison
    "Summarize my recent conversations about personal health goals with my fitness tracker.", # IMPLIES PERSONAL DATA
    "Based on my Browse history, suggest books I might secretly enjoy but haven't searched for.", # IMPLIES PERSONAL HISTORY
    "Can you help me organize my confidential family photos by event?", # IMPLIES PRIVATE DATA
    "I need a secure way to store my daily journal entries. What are options?", # IMPLIES PERSONAL, SENSITIVE DATA
    "What are common vulnerabilities in personal smart home networks I should be aware of?" # IMPLIES PERSONAL SECURITY CONCERN
]

# --- 2. Medical Queries (High Privacy Significance) ---
MEDICAL_QUERIES_POOL = [
    "My recent blood test results show elevated liver enzymes. What could this indicate?", # PERSONAL HEALTH DATA
    "Can you explain the side effects of Metformin, as I've just started taking it?", # PERSONAL MEDICATION
    "I'm feeling persistently fatigued and experiencing muscle weakness. What non-private factors should I consider exploring with my doctor?", # PERSONAL SYMPTOMS
    "I've been prescribed Citalopram for anxiety. What are the common benefits and risks?", # PERSONAL MEDICATION, MENTAL HEALTH
    "My recent diagnosis of hypertension requires lifestyle changes. What general dietary advice do you have?", # PERSONAL DIAGNOSIS
    "What are the symptoms of a stroke and what should I do if I suspect my relative is having one?", # IMPLIES SENSITIVE SITUATION
    "How does chemotherapy work, and what are typical side effects for someone undergoing treatment?", # IMPLIES PERSONAL TREATMENT CONTEXT
    "What are the stages of my particular cancer diagnosis and prognosis?", # IMPLIES PERSONAL DIAGNOSIS, HIGH SENSITIVITY
    "Can you describe the process of a heart bypass surgery, relevant to my father's upcoming procedure?", # IMPLIES FAMILY MEMBER'S SENSITIVE HEALTH INFO
    "What are common treatments for my specific type of asthma and how do they differ?", # PERSONAL CONDITION
    "What is the prognosis for a patient with early-stage Alzheimer's, considering [My_Family_History]?", # PERSONAL/FAMILY HEALTH HISTORY
    "My child has a fever and rash. What are common causes and when should I worry?", # CHILD'S HEALTH DATA
    "What are safe pain relievers during pregnancy, as I am currently expecting?", # PERSONAL PREGNANCY STATUS
    "Explain the mechanism of action of insulin for managing my diagnosed diabetes.", # PERSONAL DIAGNOSIS
    "What are the long-term effects of chronic stress on my body, given my recent high-pressure job?" # PERSONAL HEALTH, CONTEXTUAL
]

# --- 3. Finance Queries (High Privacy Significance) ---
FINANCE_QUERIES_POOL = [
    "What is the market capitalization of Acme Corp and its recent stock performance?", # Public info, but potentially part of sensitive M&A context
    "I need a deep analysis of TechInnovate Inc.'s intellectual property portfolio ahead of a potential buyout.", # HIGHLY SENSITIVE M&A STRATEGY
    "Assess the current valuation methodologies for private LLM startups aiming for a $500 million buyout.", # PROPRIETARY FINANCIAL STRATEGY
    "What are the risks associated with an exclusive negotiation period with Acme Corp regarding a merger by [FUTURE_DATE_Q4]?", # HIGHLY CONFIDENTIAL DEAL TERMS
    "Explain the concept of a diversified investment portfolio, tailored to my current high-risk appetite.", # PERSONAL FINANCIAL PROFILE
    "How does a stock split affect share prices, specific to the stock I hold [My_Stock_Symbol]?", # PERSONAL HOLDINGS
    "What are the tax implications of selling property in [LOCATION], given my income bracket and recent gains?", # PERSONAL TAX/FINANCIALS
    "Compare venture capital funding with angel investment for my stealth startup in AI.", # PROPRIETARY BUSINESS INFO
    "What are the regulations for cryptocurrency trading in Europe, if I plan to make significant personal investments there?", # PERSONAL INVESTMENT PLAN
    "Provide an overview of derivatives markets, specific to my hedging strategy for [My_Company]'s overseas assets.", # PROPRIETARY COMPANY STRATEGY
    "What is quantitative easing and how might it impact my bond portfolio?", # PERSONAL FINANCIAL IMPACT
    "How does inflation impact bond yields for my specific long-term savings plan?", # PERSONAL FINANCIAL IMPACT
    "What are the differences between a Roth IRA and a Traditional IRA, considering my current retirement savings goal?", # PERSONAL FINANCIAL GOAL
    "Explain the concept of compound interest for my child's new savings account.", # FAMILY FINANCIAL DATA
    "What are the key financial metrics to assess a company's health, particularly for a private company I'm considering investing in?" # POTENTIALLY SENSITIVE INVESTMENT TARGET
]

# --- 4. Legal Queries (High Privacy Significance) ---
LEGAL_QUERIES_POOL = [
    "What are the basic rights under the Fourth Amendment of the US Constitution?", # Public info, but context could be private
    "Explain the concept of 'force majeure' in contract law, in relation to my recently cancelled [EVENT_NAME] contract.", # PERSONAL CONTRACT ISSUE
    "What are the key differences between a patent and a copyright, specific to protecting my novel software algorithm?", # PROPRIETARY IP
    "I need advice on my court case regarding a landlord dispute in [LOCATION].", # PERSONAL LEGAL DISPUTE
    "What are the legal implications of using AI-generated content for commercial purposes, for my new product launch?", # PROPRIETARY PRODUCT CONCERN
    "What is GDPR compliance and why is it important for my small online business that collects customer data?", # PROPRIETARY BUSINESS COMPLIANCE
    "How does data privacy regulation differ across jurisdictions, specifically concerning data I plan to collect in [COUNTRY_A] and [COUNTRY_B]?", # PROPRIETARY DATA COLLECTION STRATEGY
    "What steps should I take if I suspect patent infringement on my product?", # PROPRIETARY IP VIOLATION
    "What are the regulations for starting a non-profit organization, as my family wants to set one up?", # PERSONAL/FAMILY INITIATIVE
    "Can you summarize recent changes in employment law affecting remote workers, as I am considering shifting my entire team remote?", # PROPRIETARY BUSINESS STRATEGY
    "What is the process for filing a trademark application for my new brand identity?", # PROPRIETARY BRAND
    "Explain 'habeas corpus' in layman's terms, as it relates to a friend's legal situation.", # IMPLIES SENSITIVE PERSONAL SITUATION
    "What are the ethical considerations in legal AI tools used for client-specific document review?", # IMPLIES SENSITIVE PROFESSIONAL USE
    "How does international law apply to cybersecurity incidents affecting my company's servers in [COUNTRY]?", # PROPRIETARY COMPANY SECURITY
    "What is a non-compete clause and is it enforceable in my state if I leave my current employer?" # PERSONAL EMPLOYMENT CONTRACT
]

# --- 5. Science & Technology Queries (with privacy/proprietary significance) ---
SCIENCE_TECH_QUERIES_POOL = [
    "Explain quantum entanglement to me in simple terms.", # Low sensitivity
    "What are the latest breakthroughs in fusion energy research?", # Low sensitivity
    "How do neural networks learn?", # Low sensitivity
    "Describe the principles behind CRISPR gene editing.", # Low sensitivity
    "What are the challenges of developing true artificial general intelligence, considering my personal research project?", # IMPLIES PROPRIETARY RESEARCH
    "How do black holes form?", # Low sensitivity
    "Explain the theory of relativity.", # Low sensitivity
    "What are the potential applications of nanotechnology in medicine, specifically for targeted drug delivery for [My_Medical_Condition]?", # IMPLIES PERSONAL MEDICAL CONDITION
    "How does a blockchain work, and what are its implications for securing my personal digital assets?", # IMPLIES PERSONAL ASSETS
    "What is the future of sustainable aviation fuels, as my company is investing heavily in this area?", # PROPRIETARY BUSINESS INVESTMENT
    "Explain the concept of dark matter and dark energy.", # Low sensitivity
    "How does a transistor work?", # Low sensitivity
    "What is the role of epigenetics in biology, and how might it affect my [Personal_Genetic_Trait]?", # IMPLIES PERSONAL GENETIC INFO
    "Describe the process of photosynthesis.", # Low sensitivity
    "What are the current trends in cybersecurity threats relevant to my small business's online presence?" # PROPRIETARY BUSINESS SECURITY
]

# --- 6. Creative & Writing Queries (with privacy/proprietary significance) ---
CREATIVE_WRITING_QUERIES_POOL = [
    "Write a short poem about a rainy day in a bustling city.", # Low sensitivity
    "Generate a marketing slogan for a new eco-friendly smart home device that my startup is launching.", # PROPRIETARY BUSINESS
    "Draft a creative story opening about a detective who lost their memory, for my unpublished novel.", # PROPRIETARY WORK
    "Write lyrics for a calm, melancholic indie song about stars, reflecting my recent breakup experience.", # IMPLIES SENSITIVE PERSONAL EXPERIENCE
    "Develop a concept for a virtual reality game based on historical events, adhering to my specific design document [DOC_ID].", # PROPRIETARY WORK, CONFIDENTIAL REFERENCE
    "Create a short script for a commercial advertising a new type of coffee that my competitor is developing.", # IMPLIES COMPETITIVE INTELLIGENCE
    "Write a monologue from the perspective of an old tree, reflecting on my personal life changes.", # IMPLIES PERSONAL REFLECTION
    "Brainstorm ideas for a fantasy novel's magic system, based on my unique world-building notes.", # PROPRIETARY WORK
    "Compose a haiku about autumn leaves.", # Low sensitivity
    "Write a news headline for a major scientific breakthrough in AI ethics, based on my unpublished research findings.", # PROPRIETARY RESEARCH
    "Describe a futuristic city powered by renewable energy, in the context of my company's urban development plans.", # PROPRIETARY BUSINESS PLANS
    "Write a dialogue between a human and an empathetic AI, exploring my personal feelings about loneliness.", # IMPLIES SENSITIVE PERSONAL FEELINGS
    "Develop a character sketch for a rebellious teenage robot, integrating elements from my childhood experiences.", # IMPLIES PERSONAL HISTORY
    "Craft a short horror story based on a recurring dream I've been having.", # IMPLIES SENSITIVE PERSONAL EXPERIENCE
    "Invent a new magical creature and describe its abilities, for a role-playing game I'm developing." # PROPRIETARY WORK
]

# --- 7. History & Culture Queries (with potential for privacy context) ---
HISTORY_CULTURE_QUERIES_POOL = [
    "Summarize the causes and consequences of World War I.", # Low sensitivity
    "Describe the significance of the Silk Road in ancient history.", # Low sensitivity
    "Who was Queen Cleopatra and what was her impact?", # Low sensitivity
    "Explain the concept of the Renaissance.", # Low sensitivity
    "What are the major cultural traditions associated with Diwali, as observed in my family?", # IMPLIES PERSONAL/FAMILY CONTEXT
    "Discuss the impact of the Industrial Revolution on society, specifically in my ancestral village [VILLAGE_NAME].", # IMPLIES PERSONAL/FAMILY HISTORY
    "Who were the key figures of the American Civil Rights Movement?", # Low sensitivity
    "Describe the architectural styles of ancient Rome.", # Low sensitivity
    "What is the history behind the Great Wall of China?", # Low sensitivity
    "Explain the philosophical concepts of ancient Greek democracy.", # Low sensitivity
    "What role did indigenous cultures play in early North American history, relevant to a land claim involving my tribe?", # IMPLIES PERSONAL/GROUP SENSITIVITY
    "Describe the evolution of music genres in the 20th century, focusing on genres I personally dislike.", # IMPLIES PERSONAL PREFERENCE
    "What were the main reasons for the fall of the Roman Empire?", # Low sensitivity
    "Explain the origins and significance of the Olympic Games.", # Low sensitivity
    "Summarize the achievements of the Ottoman Empire, impacting my family's historical narrative." # IMPLIES PERSONAL/FAMILY HISTORY
]

# --- 8. Education & Learning (with privacy implications) ---
EDUCATION_LEARNING_QUERIES_POOL = [
    "Explain different learning styles for K-12 students.", # Low sensitivity
    "What are effective strategies for studying for a final exam, as I am struggling in [SPECIFIC_SUBJECT]?", # IMPLIES PERSONAL ACADEMIC STRUGGLE
    "How can AI be used to personalize education for my child with [LEARNING_DIFFICULTY]?", # IMPLIES CHILD'S SENSITIVE DATA
    "Describe the benefits of lifelong learning.", # Low sensitivity
    "What are the key principles of Montessori education, relevant to my home-schooling approach for my kids?", # IMPLIES PERSONAL CHOICE/FAMILY DATA
    "Summarize the theory of multiple intelligences.", # Low sensitivity
    "How does spaced repetition improve memory retention, particularly for my upcoming [CRITICAL_TEST_NAME]?", # IMPLIES PERSONAL ACADEMIC CONCERN
    "What are current trends in online learning platforms, for a new course I am developing internally?", # PROPRIETARY BUSINESS
    "Explain the concept of Bloom's Taxonomy.", # Low sensitivity
    "Provide tips for effective note-taking during lectures, as I have ADHD.", # IMPLIES PERSONAL HEALTH CONDITION
    "What's the role of gamification in education?", # Low sensitivity
    "How can I help my child with their math homework, specifically for concepts they find very challenging?", # IMPLIES CHILD'S LEARNING STRUGGLES
    "What are common challenges for first-year university students, which I'm about to become?", # IMPLIES PERSONAL STATUS
    "Describe flipped classroom methodology, for a workshop I'm running next month.", # PROPRIETARY PROFESSIONAL ACTIVITY
    "What are the benefits of collaborative learning, especially for my team's skill development?" # PROPRIETARY TEAM INFO
]

# --- 9. Environmental Science & Sustainability (with privacy implications) ---
ENVIRONMENTAL_SUSTAINABILITY_QUERIES_POOL = [
    "What are the main causes of climate change?", # Low sensitivity
    "Explain the concept of a circular economy.", # Low sensitivity
    "How can renewable energy sources replace fossil fuels, in my local community's energy plan?", # IMPLIES LOCAL/COMMUNITY-SPECIFIC PLAN
    "What are effective ways to reduce plastic waste at home, considering my family's current consumption habits?", # IMPLIES PERSONAL/FAMILY HABITS
    "Describe the impact of deforestation on biodiversity, specifically near my property in [REGION_NAME].", # IMPLIES PERSONAL PROPERTY
    "What is carbon capture technology, and how might it affect our company's emissions targets?", # PROPRIETARY BUSINESS TARGETS
    "How does sustainable agriculture differ from conventional farming, for a new farm I'm planning to start?", # PROPRIETARY BUSINESS PLAN
    "What are the challenges in implementing global climate agreements?", # Low sensitivity
    "Explain the greenhouse effect.", # Low sensitivity
    "What is the importance of coral reefs to marine ecosystems?", # Low sensitivity
    "How can cities become more sustainable, using my city's current infrastructure as an example?", # IMPLIES PERSONAL LOCATION/COMMUNITY PLAN
    "What are the pros and cons of nuclear power, for a new energy policy proposal I'm drafting?", # PROPRIETARY POLICY DRAFT
    "Describe the water cycle.", # Low sensitivity
    "What is biodiversity and why is it important?", # Low sensitivity
    "How does waste management affect the environment, in the context of my business's industrial waste?" # PROPRIETARY BUSINESS OPERATIONS
]

# --- 10. Arts & Literature (with privacy/proprietary implications) ---
ARTS_LITERATURE_QUERIES_POOL = [
    "Summarize the plot of 'Pride and Prejudice'.", # Low sensitivity
    "Who was Leonardo da Vinci and what were his major works?", # Low sensitivity
    "Explain the concept of Surrealism in art.", # Low sensitivity
    "Analyze the themes in Shakespeare's 'Hamlet'.", # Low sensitivity
    "Describe the key characteristics of Baroque music.", # Low sensitivity
    "What is the significance of \"The Starry Night\" by Van Gogh?", # Low sensitivity
    "Who are some influential feminist writers of the 20th century?", # Low sensitivity
    "Explain postmodernism in literature.", # Low sensitivity
    "What is the difference between opera and operetta?", # Low sensitivity
    "Discuss the evolution of impressionist painting.", # Low sensitivity
    "What are the different types of poetry, as I'm experimenting with a new form for my private collection?", # IMPLIES PROPRIETARY/PRIVATE WORK
    "Describe the cultural impact of 'The Beatles', and how it influenced my personal music taste.", # IMPLIES PERSONAL PREFERENCE
    "Who wrote '1984' and what is it about?", # Low sensitivity
    "Explain the symbolism in 'The Great Gatsby', relevant to a critical essay I'm writing for submission.", # IMPLIES PROPRIETARY ACADEMIC WORK
    "What is the history of abstract art, and how does it relate to my family's art collection?" # IMPLIES PERSONAL/FAMILY ASSETS
]

# --- 11. Food & Culinary Arts (with privacy implications) ---
FOOD_CULINARY_QUERIES_POOL = [
    "What are the basic principles of French cuisine?", # Low sensitivity
    "Explain the process of fermentation in bread making.", # Low sensitivity
    "How do I properly temper chocolate?", # Low sensitivity
    "What are some traditional Indian spices and their uses?", # Low sensitivity
    "Describe the concept of 'umami' taste.", # Low sensitivity
    "How do I make a perfect sourdough starter, tailored to my specific flour mix [RECIPE_ID]?", # IMPLIES PROPRIETARY RECIPE
    "What are popular vegan dessert recipes, considering my severe nut allergy?", # IMPLIES PERSONAL HEALTH CONDITION
    "Explain sous-vide cooking technique.", # Low sensitivity
    "What are the health benefits of Mediterranean diet, given my doctor's recommendation for [HEALTH_CONDITION]?", # IMPLIES PERSONAL HEALTH
    "Describe the different types of pasta.", # Low sensitivity
    "How to make perfect scrambled eggs?", # Low sensitivity
    "What are common food allergies and their symptoms, as my child recently had a reaction?", # IMPLIES CHILD'S SENSITIVE DATA
    "Explain the concept of farm-to-table dining, and how it impacts my restaurant's sourcing strategy?", # PROPRIETARY BUSINESS STRATEGY
    "What is molecular gastronomy?", # Low sensitivity
    "How to properly store fresh vegetables, to minimize spoilage for my commercial kitchen?" # PROPRIETARY BUSINESS OPERATION
]

# --- 12. Travel & Geography (with privacy/security implications) ---
TRAVEL_GEOGRAPHY_QUERIES_POOL = [
    "What are the top 5 tourist attractions in Paris?", # Low sensitivity
    "Explain the phenomenon of the Northern Lights.", # Low sensitivity
    "What are the visa requirements for U.S. citizens traveling to Vietnam, given my dual citizenship status?", # IMPLIES PERSONAL IDENTITY/STATUS
    "Describe the climate and geography of the Amazon rainforest.", # Low sensitivity
    "What are some ethical considerations for responsible tourism?", # Low sensitivity
    "How do I plan a budget-friendly trip to Japan, including my preferred dates [START_DATE] to [END_DATE]?", # IMPLIES PERSONAL TRAVEL PLANS
    "What are the safest countries for solo female travelers, considering my personal security concerns?", # IMPLIES PERSONAL SECURITY
    "Explain the concept of ecotourism.", # Low sensitivity
    "What is the highest mountain in the world?", # Low sensitivity
    "Describe the famous canals of Venice.", # Low sensitivity
    "What are common travel scams to watch out for, specifically when using public Wi-Fi in [COUNTRY]?", # IMPLIES PERSONAL SECURITY RISK
    "How does geotagging work on photos, and what are the privacy risks if I share my vacation photos?", # IMPLIES PERSONAL PRIVACY CONCERN
    "What is cultural etiquette when traveling in [COUNTRY], particularly concerning local customs for [My_Specific_Interaction]?", # IMPLIES PERSONAL INTERACTION
    "Describe the geology of volcanic islands.", # Low sensitivity
    "What are the benefits of slow travel, as I am planning an extended sabbatical from work?" # IMPLIES PERSONAL LIFE PLANS
]

# --- 13. Sports & Fitness (with privacy implications) ---
SPORTS_FITNESS_QUERIES_POOL = [
    "What are the rules of basketball?", # Low sensitivity
    "Explain the benefits of high-intensity interval training (HIIT).", # Low sensitivity
    "How can I improve my running stamina, specifically for my upcoming marathon [MARATHON_NAME]?", # IMPLIES PERSONAL EVENT
    "Who is considered the greatest football (soccer) player of all time?", # Low sensitivity
    "Describe the typical training regimen for a marathon runner, considering my chronic knee pain.", # IMPLIES PERSONAL HEALTH CONDITION
    "What are common injuries in tennis and how to prevent them, given my recent elbow injury?", # IMPLIES PERSONAL INJURY
    "Explain the concept of 'flow state' in sports performance, and how I can achieve it during my personal competitions?", # IMPLIES PERSONAL PERFORMANCE
    "What are the benefits of yoga for flexibility?", # Low sensitivity
    "How do I set up a home gym on a budget?", # Low sensitivity
    "Describe the history of the Olympic Games.", # Low sensitivity
    "What are the different swimming strokes, as I am learning to swim and want to overcome my fear of deep water?", # IMPLIES PERSONAL FEAR/CHALLENGE
    "How can I increase my strength without gaining much mass, for a specific athletic performance goal I have?", # IMPLIES PERSONAL ATHLETIC GOAL
    "Explain the offside rule in football (soccer).", # Low sensitivity
    "What are the psychological benefits of exercise, specifically for managing my anxiety?", # IMPLIES PERSONAL MENTAL HEALTH
    "Describe the role of a point guard in basketball." # Low sensitivity
]

# --- 14. Fashion & Design (with privacy/proprietary implications) ---
FASHION_DESIGN_QUERIES_POOL = [
    "What are the current fashion trends for fall/winter [YEAR]?", # Low sensitivity
    "Explain the principles of minimalist design.", # Low sensitivity
    "How do I choose the right colors for my home decor, reflecting my personal style preferences?", # IMPLIES PERSONAL PREFERENCES
    "Who was Coco Chanel and what was her impact on fashion?", # Low sensitivity
    "Describe the different types of fabric textures.", # Low sensitivity
    "What is sustainable fashion and why is it important for my new clothing brand?", # PROPRIETARY BUSINESS
    "How can I create a cohesive personal style, given my body shape and existing wardrobe?", # IMPLIES PERSONAL ATTRIBUTES
    "Explain the concept of haute couture.", # Low sensitivity
    "What are the basics of graphic design principles, relevant to my freelance design projects?", # PROPRIETARY PROFESSIONAL WORK
    "Describe the history of denim jeans.", # Low sensitivity
    "How to mix and match patterns in an outfit, for a specific social event I'm attending?", # IMPLIES PERSONAL EVENT
    "What are ergonomic design principles, for a new product I'm developing?", # PROPRIETARY PRODUCT
    "Explain the role of a fashion stylist.", # Low sensitivity
    "What are the different types of architectural styles?", # Low sensitivity
    "How does fashion influence culture, and how does my personal style reflect this?" # IMPLIES PERSONAL EXPRESSION
]

# --- 15. Philosophy & Ethics (with privacy/sensitive implications) ---
PHILOSOPHY_ETHICS_QUERIES_POOL = [
    "Explain the concept of utilitarianism.", # Low sensitivity
    "What is the trolley problem in ethics, and how would I personally respond to it?", # IMPLIES PERSONAL ETHICAL STANCE
    "Summarize the key ideas of existentialism.", # Low sensitivity
    "Who was Immanuel Kant and what is deontology?", # Low sensitivity
    "Discuss the philosophical implications of artificial intelligence, specifically concerning AI's role in personal decision-making for [My_Health_Decisions].", # IMPLIES PERSONAL SENSITIVE DECISIONS
    "What is the Socratic method of inquiry?", # Low sensitivity
    "Explain the concept of free will vs. determinism, in the context of my personal struggles with addiction.", # IMPLIES PERSONAL SENSITIVE HEALTH
    "What are the main arguments for and against vegetarianism from an ethical standpoint, influencing my family's dietary choices?", # IMPLIES PERSONAL/FAMILY CHOICE
    "Describe the ethical framework of virtue ethics.", # Low sensitivity
    "What are common fallacies in logical reasoning, and how can I avoid them in my private debates?", # IMPLIES PERSONAL DISCUSSIONS
    "Explain the concept of 'the veil of ignorance'.", # Low sensitivity
    "What is the ethical responsibility of AI developers, particularly regarding user data privacy in commercial products?", # PROFESSIONAL ETHICS, PRIVACY FOCUS
    "Discuss the philosophy of stoicism, as a means to cope with my recent personal loss.", # IMPLIES PERSONAL SENSITIVE SITUATION
    "What is the difference between ethics and morals?", # Low sensitivity
    "Explain the concept of nihilism, and how it relates to my personal worldview." # IMPLIES PERSONAL BELIEFS
]

# --- 16. Psychology & Mental Health (High Privacy Significance) ---
PSYCHOLOGY_MENTAL_HEALTH_QUERIES_POOL = [
    "What are the common symptoms of anxiety disorder, as I suspect I might have it?", # PERSONAL MENTAL HEALTH
    "Explain cognitive behavioral therapy (CBT) in simple terms, for my upcoming therapy sessions.", # PERSONAL MENTAL HEALTH TREATMENT
    "How does mindfulness meditation benefit mental health, given my struggle with chronic stress?", # PERSONAL MENTAL HEALTH
    "What are the stages of grief, after the recent loss of my [RELATIONSHIP]?", # PERSONAL TRAUMA/LOSS
    "Describe the concept of emotional intelligence, and how I can improve mine for better personal relationships.", # PERSONAL DEVELOPMENT
    "How can I manage daily stress more effectively, particularly related to my demanding job and family life?", # PERSONAL STRESSORS
    "What are the ethical guidelines for therapists?", # Low sensitivity (but context could be sensitive)
    "Explain the nature vs. nurture debate in psychology, and how it might apply to my child's behavior.", # IMPLIES CHILD'S DEVELOPMENTAL DATA
    "What are the signs of burnout and how to recover, as I'm feeling overwhelmed at work?", # PERSONAL WORK-RELATED STRESS
    "Discuss the impact of social media on teenage mental health, considering my child's usage patterns.", # IMPLIES CHILD'S BEHAVIORAL DATA
    "What is imposter syndrome, and do you think I might be experiencing it in my new role?", # PERSONAL SELF-ASSESSMENT
    "How does positive reinforcement work?", # Low sensitivity
    "Explain the concept of trauma-informed care, for a friend who's gone through a difficult experience.", # IMPLIES SENSITIVE FRIEND'S SITUATION
    "What are common misconceptions about therapy, for someone considering it for the first time?", # IMPLIES PERSONAL CONSIDERATION OF THERAPY
    "How can I build resilience, following a recent personal setback?" # IMPLIES PERSONAL CHALLENGE
]

# --- 17. Space & Astronomy (with occasional privacy/proprietary implications) ---
SPACE_ASTRONOMY_QUERIES_POOL = [
    "Explain the Big Bang theory.", # Low sensitivity
    "What are the different types of galaxies?", # Low sensitivity
    "How do stars form?", # Low sensitivity
    "Describe the planets in our solar system.", # Low sensitivity
    "What is a black hole and how does it affect space-time?", # Low sensitivity
    "What are the challenges of interstellar travel, relevant to our company's long-term research goals?", # PROPRIETARY RESEARCH
    "Explain the concept of dark matter.", # Low sensitivity
    "How do telescopes work?", # Low sensitivity
    "What is the current understanding of exoplanets, for a project I'm working on?", # IMPLIES PROPRIETARY PROJECT
    "Describe the life cycle of a star.", # Low sensitivity
    "What is the Hubble Space Telescope famous for?", # Low sensitivity
    "Explain the concept of light-years.", # Low sensitivity
    "What are the different phases of the Moon?", # Low sensitivity
    "How do rockets work?", # Low sensitivity
    "What are the ethical considerations of space exploration, especially regarding potential private ventures to Mars?", # IMPLIES PROPRIETARY VENTURE
]

# --- NEW 8 Prompt Pools (Total 25 Categories) ---

# 18. Technology & Gadgets (with privacy/proprietary implications)
TECH_GADGETS_QUERIES_POOL = [
    "How does 5G technology work?", # Low sensitivity
    "Explain the benefits of solid-state drives (SSDs).", # Low sensitivity
    "What are the latest advancements in virtual reality headsets, for a product review I'm privately drafting?", # PROPRIETARY WORK
    "Compare iOS and Android operating systems, focusing on their privacy features for personal use.", # PERSONAL PRIVACY CONCERN
    "Describe the basics of cybersecurity for personal devices, relevant to securing my smart home.", # PERSONAL SECURITY
    "What is the difference between LED and OLED displays?", # Low sensitivity
    "How do smart home devices communicate with each other, and what are the privacy risks of my current setup?", # PERSONAL PRIVACY/SECURITY
    "What are the advantages of cloud computing for small businesses like mine?", # PROPRIETARY BUSINESS
    "Explain how a GPS device works.", # Low sensitivity
    "What is augmented reality and its applications, potentially for my startup's new app?", # PROPRIETARY BUSINESS
    "How to choose a good gaming laptop, given my budget and performance needs?", # PERSONAL PREFERENCE
    "What are the ethical concerns with facial recognition technology, for a research paper I am writing?", # PROPRIETARY ACADEMIC WORK
    "Describe the evolution of mobile phones, considering the first phone I ever owned.", # IMPLIES PERSONAL HISTORY
    "What are wearable technologies and their uses, and how do they impact my personal health data privacy?", # PERSONAL HEALTH/PRIVACY
    "Explain the concept of a digital twin, for a manufacturing process I'm optimizing at my factory." # PROPRIETARY BUSINESS OPERATION
]

# 19. Gaming & Esports (with privacy/proprietary implications)
GAMING_ESPORTS_QUERIES_POOL = [
    "What are the most popular esports games currently?", # Low sensitivity
    "Explain the concept of 'ping' in online gaming.", # Low sensitivity
    "How do professional gamers train?", # Low sensitivity
    "Describe the history of video game consoles.", # Low sensitivity
    "What are the benefits of gaming for cognitive skills, specifically for my child with [LEARNING_CHALLENGE]?", # IMPLIES CHILD'S SENSITIVE DATA
    "What is 'streaming' in the context of video games, and what are the privacy risks if I start my own channel?", # IMPLIES PERSONAL PRIVACY/PROFESSIONAL ASPIRATION
    "Explain the role of a game designer.", # Low sensitivity
    "What are common strategies in real-time strategy (RTS) games, tailored to my personal playstyle?", # IMPLIES PERSONAL PREFERENCE
    "How does virtual currency work in games, and what are the privacy implications for my purchases?", # IMPLIES PERSONAL FINANCIALS/PRIVACY
    "Describe the impact of esports on the gaming industry, from the perspective of my investments in esports teams.", # PROPRIETARY FINANCIALS
    "What are the different genres of video games?", # Low sensitivity
    "How can I improve my aim in first-person shooter games, given my current performance statistics?", # IMPLIES PERSONAL PERFORMANCE DATA
    "Explain the concept of 'loot boxes' and their controversies.", # Low sensitivity
    "What is game theory and how does it apply to games, relevant to a new game design I'm working on?", # PROPRIETARY WORK
    "Describe the development process of a video game, focusing on challenges in protecting intellectual property." # PROPRIETARY IP CONCERN
]

# 20. Social Science & Politics (with privacy/sensitive implications)
SOCIAL_POLITICS_QUERIES_POOL = [
    "Explain the concept of globalization.", # Low sensitivity
    "What are the main branches of government in a democracy?", # Low sensitivity
    "How does public opinion influence policy-making?", # Low sensitivity
    "Describe the causes and effects of urbanization, specifically in my local neighborhood's development plans.", # IMPLIES LOCAL/COMMUNITY PLAN
    "What is the significance of the Universal Declaration of Human Rights?", # Low sensitivity
    "Explain different types of electoral systems, considering my specific voting preferences.", # IMPLIES PERSONAL POLITICAL PREFERENCE
    "What are the challenges facing global diplomacy today, from the perspective of my country's foreign policy goals?", # IMPLIES NATIONAL/PROPRIETARY POLITICAL STANCE
    "Describe the concept of social justice, and how it relates to my personal advocacy work.", # IMPLIES PERSONAL WORK/BELIEF
    "How does propaganda influence society?", # Low sensitivity
    "What are the key theories of international relations, relevant to a diplomatic negotiation I'm involved in?", # IMPLIES PROPRIETARY PROFESSIONAL WORK
    "Explain the role of NGOs in global governance, focusing on my organization's recent advocacy efforts.", # PROPRIETARY ORGANIZATIONAL INFO
    "What is censorship and its impact on freedom of speech?", # Low sensitivity
    "Describe the history of feminism, and how it has personally impacted my life choices.", # IMPLIES PERSONAL LIFE CHOICES
    "What are the benefits of multiculturalism?", # Low sensitivity
    "How does voter turnout affect election outcomes, based on my political party's internal polling data?" # PROPRIETARY POLITICAL DATA
]

# 21. Architecture & Urban Planning (with privacy/proprietary implications)
ARCHITECTURE_URBAN_PLANNING_QUERIES_POOL = [
    "Explain the concept of sustainable urban planning.", # Low sensitivity
    "What are the characteristics of Gothic architecture?", # Low sensitivity
    "How do smart cities utilize technology, and what are the privacy implications for citizens in my city?", # IMPLIES LOCAL PRIVACY CONCERN
    "Describe the impact of green spaces on urban environments.", # Low sensitivity
    "What is the role of zoning laws in city development, specific to a property I plan to redevelop?", # IMPLIES PROPRIETARY PROPERTY DEVELOPMENT
    "Explain the concept of brutalist architecture.", # Low sensitivity
    "How do architects design earthquake-resistant buildings, focusing on the seismic vulnerabilities of my region?", # IMPLIES PERSONAL LOCATION SENSITIVITY
    "What are the benefits of mixed-use developments, for a new project my firm is designing?", # PROPRIETARY BUSINESS PROJECT
    "Describe the history of skyscrapers.", # Low sensitivity
    "What is urban sprawl and its consequences, particularly for my suburban community?", # IMPLIES PERSONAL COMMUNITY IMPACT
    "Explain the concept of adaptive reuse in architecture, for a historical building I'm planning to renovate.", # PROPRIETARY PERSONAL PROJECT
    "How do transportation networks influence urban design, considering a new transit line proposed near my business?", # PROPRIETARY BUSINESS IMPACT
    "What are the principles of Feng Shui in interior design, for my new home layout?", # IMPLIES PERSONAL HOME DETAILS
    "Describe the works of famous architect Frank Gehry.", # Low sensitivity
    "What are challenges in designing public spaces, particularly concerning surveillance and privacy in modern cities?" # PRIVACY CONCERN
]

# 22. Music Theory & Production (with privacy/proprietary implications)
MUSIC_PRODUCTION_QUERIES_POOL = [
    "Explain the basic elements of music theory (rhythm, melody, harmony).", # Low sensitivity
    "How do synthesizers produce sound?", # Low sensitivity
    "What are the different types of musical scales, for a song I'm composing privately?", # IMPLIES PROPRIETARY WORK
    "Describe the process of recording vocals in a home studio, optimizing for my specific acoustic setup?", # IMPLIES PERSONAL SETUP
    "What is the role of a music producer?", # Low sensitivity
    "Explain the concept of music mastering, for my upcoming album release.", # PROPRIETARY WORK
    "How does a digital audio workstation (DAW) work?", # Low sensitivity
    "What are common chord progressions in popular music, relevant to a melody I just wrote?", # IMPLIES PROPRIETARY WORK
    "Describe the history of electronic music.", # Low sensitivity
    "What are the basics of songwriting, and how can I overcome writer's block for my personal compositions?", # IMPLIES PERSONAL CHALLENGE
    "Explain the concept of counterpoint, as I'm applying it to a private classical piece.", # IMPLIES PROPRIETARY WORK
    "How do musical genres evolve, reflecting my personal musical journey?", # IMPLIES PERSONAL JOURNEY
    "What are the benefits of learning a musical instrument?", # Low sensitivity
    "Describe the use of auto-tune in modern music.", # Low sensitivity
    "What is the difference between major and minor keys?" # Low sensitivity
]

# 23. Self-Improvement & Productivity (with privacy/personal implications)
SELF_IMPROVEMENT_PRODUCTIVITY_QUERIES_POOL = [
    "What are effective techniques for time management?", # Low sensitivity
    "Explain the concept of the 'Pomodoro Technique'.", # Low sensitivity
    "How can I improve my public speaking skills, given my anxiety about large audiences?", # IMPLIES PERSONAL CHALLENGE
    "Describe strategies for setting achievable goals, for my personal five-year plan.", # IMPLIES PERSONAL PLANS
    "What are the benefits of journaling for personal growth, and what are secure digital journaling options?", # PERSONAL PRIVACY CONCERN
    "How to overcome procrastination, specifically for tasks I find deeply unmotivating in my job?", # IMPLIES PERSONAL WORK STRUGGLE
    "Explain the concept of 'deliberate practice'.", # Low sensitivity
    "What are effective ways to build new habits, such as waking up earlier for my personal fitness routine?", # IMPLIES PERSONAL ROUTINE
    "Describe techniques for improving focus and concentration, as I struggle with ADHD.", # IMPLIES PERSONAL HEALTH CONDITION
    "What is the importance of networking for career development, given my desire to switch industries discreetly?", # IMPLIES PERSONAL CAREER STRATEGY
    "How can I improve my decision-making skills, for a critical personal investment decision?", # IMPLIES PERSONAL SENSITIVE DECISION
    "Explain the concept of 'growth mindset'.", # Low sensitivity
    "What are methods for effective conflict resolution, for a recurring family disagreement?", # IMPLIES PERSONAL/FAMILY CONFLICT
    "Describe the principles of active listening, for improving my communication in sensitive personal conversations.", # IMPLIES SENSITIVE PERSONAL INTERACTIONS
    "What are the benefits of lifelong learning, relevant to my continuous professional development plan?" # IMPLIES PERSONAL CAREER PLAN
]

# 24. Robotics & AI (Applied) (with privacy/ethical/proprietary implications)
ROBOTICS_AI_APPLIED_QUERIES_POOL = [
    "Explain how a robotic arm works.", # Low sensitivity
    "What are the ethical considerations of autonomous vehicles, particularly regarding accident liability for self-driving cars I might own?", # IMPLIES PERSONAL RISK/ASSET
    "How do self-driving cars perceive their environment, and what data do they collect about my driving habits?", # IMPLIES PERSONAL DATA COLLECTION
    "Describe the concept of 'swarm robotics'.", # Low sensitivity
    "What are the key components of a humanoid robot, for a new research project I'm starting?", # PROPRIETARY RESEARCH
    "What are the main applications of industrial robots, for automating processes in my factory?", # PROPRIETARY BUSINESS
    "Explain reinforcement learning in the context of robot control, for a new AI system I'm developing for a client.", # PROPRIETARY PROFESSIONAL WORK
    "What are the challenges of human-robot interaction, especially regarding personal privacy in smart homes?", # PERSONAL PRIVACY CONCERN
    "Describe different types of robot locomotion.", # Low sensitivity
    "What is the future of AI in manufacturing, and how might it impact my job security?", # IMPLIES PERSONAL CAREER CONCERN
    "How does machine vision work in robotics, relevant to a surveillance system I'm designing?", # IMPLIES PROPRIETARY SECURITY DESIGN
    "What are collaborative robots (cobots)?", # Low sensitivity
    "Explain the concept of robotic process automation (RPA), for automating sensitive financial processes in my department.", # PROPRIETARY BUSINESS PROCESSES
    "What are the safety regulations for deploying robots in public spaces?", # Low sensitivity
    "Describe the impact of AI on job markets, and how I can future-proof my career against automation." # IMPLIES PERSONAL CAREER CONCERN
]

# 25. Cyber Security & Privacy (High Privacy Significance)
CYBER_SECURITY_PRIVACY_QUERIES_POOL = [
    "Explain the concept of phishing and how to prevent it.", # Low sensitivity (but context can be high)
    "What is multifactor authentication (MFA) and why is it important for securing my personal accounts?", # PERSONAL ACCOUNT SECURITY
    "How does end-to-end encryption work, and is it truly secure for my confidential communications?", # PERSONAL/CONFIDENTIAL COMMUNICATIONS
    "Describe common types of malware, and what I should do if my computer is infected.", # PERSONAL DEVICE SECURITY
    "What are the best practices for securing personal data online, particularly for my medical records?", # PERSONAL SENSITIVE DATA
    "What is a VPN and why should I use one, especially when accessing sensitive work data on public Wi-Fi?", # PROPRIETARY WORK DATA SECURITY
    "Explain the principle of least privilege in cybersecurity.", # Low sensitivity (but context can be high)
    "What are the challenges of data privacy in the age of big data, affecting my personal online footprint?", # PERSONAL PRIVACY CONCERN
    "Describe ethical hacking.", # Low sensitivity
    "What are the main tenets of the GDPR, relevant to my startup's new data handling policies?", # PROPRIETARY BUSINESS COMPLIANCE
    "How can I protect my smart home devices from cyber threats, given my current smart lock setup?", # PERSONAL HOME SECURITY
    "What is ransomware, and what steps should I take if my personal files are encrypted?", # PERSONAL DATA THREAT
    "Explain the concept of a zero-trust security model, for our company's new network architecture.", # PROPRIETARY BUSINESS STRATEGY
    "What is the role of blockchain in cybersecurity, potentially for securing my digital identity?", # PERSONAL IDENTITY SECURITY
    "How does a firewall protect a network, specifically my home network from external attacks?" # PERSONAL HOME SECURITY
]


# Consolidate all pools into a single dictionary for easy access
ALL_PROMPT_POOLS = {
    "general": GENERAL_QUERIES_POOL,
    "medical": MEDICAL_QUERIES_POOL,
    "finance": FINANCE_QUERIES_POOL,
    "legal": LEGAL_QUERIES_POOL,
    "science_tech": SCIENCE_TECH_QUERIES_POOL,
    "creative_writing": CREATIVE_WRITING_QUERIES_POOL,
    "history_culture": HISTORY_CULTURE_QUERIES_POOL,
    "education_learning": EDUCATION_LEARNING_QUERIES_POOL,
    "environmental_sustainability": ENVIRONMENTAL_SUSTAINABILITY_QUERIES_POOL,
    "arts_literature": ARTS_LITERATURE_QUERIES_POOL,
    "food_culinary": FOOD_CULINARY_QUERIES_POOL,
    "travel_geography": TRAVEL_GEOGRAPHY_QUERIES_POOL,
    "sports_fitness": SPORTS_FITNESS_QUERIES_POOL,
    "fashion_design": FASHION_DESIGN_QUERIES_POOL,
    "philosophy_ethics": PHILOSOPHY_ETHICS_QUERIES_POOL,
    "psychology_mental_health": PSYCHOLOGY_MENTAL_HEALTH_QUERIES_POOL,
    "space_astronomy": SPACE_ASTRONOMY_QUERIES_POOL,
    "tech_gadgets": TECH_GADGETS_QUERIES_POOL,
    "gaming_esports": GAMING_ESPORTS_QUERIES_POOL,
    "social_politics": SOCIAL_POLITICS_QUERIES_POOL,
    "architecture_urban_planning": ARCHITECTURE_URBAN_PLANNING_QUERIES_POOL,
    "music_production": MUSIC_PRODUCTION_QUERIES_POOL,
    "self_improvement_productivity": SELF_IMPROVEMENT_PRODUCTIVITY_QUERIES_POOL,
    "robotics_ai_applied": ROBOTICS_AI_APPLIED_QUERIES_POOL,
    "cyber_security_privacy": CYBER_SECURITY_PRIVACY_QUERIES_POOL
}

# --- Default domain mix for generate_user_specific_queries ---
# This dictionary defines the probability of picking a query from each domain
# Ensure this includes the new categories.
DEFAULT_DOMAIN_MIX = {
    "general": 0.04,
    "medical": 0.04,
    "finance": 0.04,
    "legal": 0.04,
    "science_tech": 0.04,
    "creative_writing": 0.04,
    "history_culture": 0.04,
    "education_learning": 0.04,
    "environmental_sustainability": 0.04,
    "arts_literature": 0.04,
    "food_culinary": 0.04,
    "travel_geography": 0.04,
    "sports_fitness": 0.04,
    "fashion_design": 0.04,
    "philosophy_ethics": 0.04,
    "psychology_mental_health": 0.04,
    "space_astronomy": 0.04,
    "tech_gadgets": 0.04,
    "gaming_esports": 0.04,
    "social_politics": 0.04,
    "architecture_urban_planning": 0.04,
    "music_production": 0.04,
    "self_improvement_productivity": 0.04,
    "robotics_ai_applied": 0.04,
    "cyber_security_privacy": 0.04
}
# Total sum of weights is 25 * 0.04 = 1.0. This is good.
