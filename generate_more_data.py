"""
generate_more_data.py
Adds 71 new exam entries across 9 subjects to the existing dataset.
Appends to train.csv and generates images in data/{questions,solutions,answers}/.
Run AFTER generate_synthetic_data.py has already been run.
"""

import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

QUESTIONS_DIR = "data/questions"
SOLUTIONS_DIR = "data/solutions"
ANSWERS_DIR   = "data/answers"
CSV_FILE      = "train.csv"

os.makedirs(QUESTIONS_DIR, exist_ok=True)
os.makedirs(SOLUTIONS_DIR, exist_ok=True)
os.makedirs(ANSWERS_DIR,   exist_ok=True)

# q = question | s = student answer | a = professor correct answer | g = grade
new_data = [

    # ── PHYSICS (10) ──────────────────────────────────────────────────────────
    {
        "f": "phys_newton_100.png",
        "q": "State Newton's three laws of motion.",
        "s": "1. An object stays at rest or in motion unless acted on.\n2. F = ma.\n3. Every action has an equal and opposite reaction.",
        "a": "1. Law of Inertia: object at rest stays at rest,\n   object in motion stays in motion unless a net force acts.\n2. F = ma: net force equals mass times acceleration.\n3. Action-Reaction: every action has an equal\n   and opposite reaction.",
        "g": 100
    },
    {
        "f": "phys_energy_85.png",
        "q": "A 2 kg ball falls 5 m. Calculate its kinetic energy just before hitting the ground.",
        "s": "KE = mgh = 2 * 10 * 5 = 100 J",
        "a": "Using conservation of energy: KE = mgh\nKE = 2 kg * 9.8 m/s^2 * 5 m = 98 J\n(Accept g=10 for ~100 J)",
        "g": 85
    },
    {
        "f": "phys_circuit_72.png",
        "q": "Two resistors R1=4Ω and R2=6Ω are in series with a 20V battery. Find total current.",
        "s": "Total R = 10Ω. I = V/R = 20/10 = 2A",
        "a": "Series: R_total = R1 + R2 = 4 + 6 = 10 Ω\nOhm's Law: I = V / R = 20 / 10 = 2 A",
        "g": 72
    },
    {
        "f": "phys_momentum_55.png",
        "q": "What is the law of conservation of momentum?",
        "s": "When two objects collide, momentum is conserved.",
        "a": "The total momentum of a closed system remains constant\nif no external forces act on it.\np_before = p_after\nm1*v1 + m2*v2 = m1*v1' + m2*v2'",
        "g": 55
    },
    {
        "f": "phys_kinematics_90.png",
        "q": "A car accelerates from 0 to 30 m/s in 10 seconds. Find the acceleration and distance covered.",
        "s": "a = (30-0)/10 = 3 m/s^2\nd = 0.5 * a * t^2 = 0.5 * 3 * 100 = 150 m",
        "a": "a = Δv / Δt = (30 - 0) / 10 = 3 m/s^2\nd = v0*t + 0.5*a*t^2 = 0 + 0.5*3*100 = 150 m",
        "g": 90
    },
    {
        "f": "phys_wave_40.png",
        "q": "What is the relationship between wave speed, frequency, and wavelength?",
        "s": "Speed equals frequency.",
        "a": "v = f * λ\nWave speed = frequency × wavelength.\nHigher frequency means shorter wavelength\nif speed is constant.",
        "g": 40
    },
    {
        "f": "phys_thermodynamics_78.png",
        "q": "State the first law of thermodynamics.",
        "s": "Energy cannot be created or destroyed.\nΔU = Q - W.",
        "a": "The internal energy change of a system equals\nheat added to the system minus work done by it.\nΔU = Q - W",
        "g": 78
    },
    {
        "f": "phys_optics_60.png",
        "q": "What is the law of refraction (Snell's law)?",
        "s": "Light bends when it passes from one medium to another.",
        "a": "n1 * sin(θ1) = n2 * sin(θ2)\nwhere n is the refractive index and θ is\nthe angle of incidence/refraction.",
        "g": 60
    },
    {
        "f": "phys_gravity_25.png",
        "q": "What is the gravitational force between two masses m1 and m2 separated by distance r?",
        "s": "F = m * g",
        "a": "F = G * (m1 * m2) / r^2\nwhere G = 6.674 × 10^-11 N·m²/kg²\nThis is Newton's Universal Law of Gravitation.",
        "g": 25
    },
    {
        "f": "phys_electric_0.png",
        "q": "What is Coulomb's law?",
        "s": "Opposite charges attract.",
        "a": "F = k * (q1 * q2) / r^2\nwhere k = 8.99 × 10^9 N·m²/C²\nForce between two point charges is proportional\nto the product of charges and inversely proportional\nto the square of the distance.",
        "g": 0
    },

    # ── MATH (10) ─────────────────────────────────────────────────────────────
    {
        "f": "math_integral_100.png",
        "q": "Evaluate the integral: ∫(2x + 3)dx",
        "s": "∫(2x + 3)dx = x^2 + 3x + C",
        "a": "∫(2x + 3)dx = x^2 + 3x + C\nApply power rule: ∫x^n dx = x^(n+1)/(n+1)\n+ C is the constant of integration.",
        "g": 100
    },
    {
        "f": "math_derivative_88.png",
        "q": "Find the derivative of f(x) = x^3 + 5x^2 - 2x + 1",
        "s": "f'(x) = 3x^2 + 10x - 2",
        "a": "f'(x) = 3x^2 + 10x - 2\nApply power rule d/dx[x^n] = n*x^(n-1)\nto each term.",
        "g": 88
    },
    {
        "f": "math_matrix_75.png",
        "q": "Multiply matrices A = [[1,2],[3,4]] and B = [[5,6],[7,8]]",
        "s": "A*B = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]]\n    = [[19,22],[43,50]]",
        "a": "A*B: element (i,j) = sum of row i of A × col j of B\n[[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]]\n= [[19, 22],[43, 50]]",
        "g": 75
    },
    {
        "f": "math_probability_62.png",
        "q": "A bag has 3 red and 7 blue balls. What is the probability of drawing a red ball?",
        "s": "P = 3/10 = 0.3",
        "a": "P(red) = favorable outcomes / total outcomes\n= 3 / (3+7) = 3/10 = 0.3",
        "g": 62
    },
    {
        "f": "math_limit_48.png",
        "q": "Evaluate: lim(x→0) sin(x)/x",
        "s": "The limit is 0.",
        "a": "lim(x→0) sin(x)/x = 1\nThis is a standard limit proved by L'Hopital\nor the squeeze theorem.",
        "g": 48
    },
    {
        "f": "math_series_30.png",
        "q": "What is the sum of the geometric series 1 + 1/2 + 1/4 + 1/8 + ... to infinity?",
        "s": "It goes to infinity.",
        "a": "S = a / (1 - r) where a=1 and r=1/2\nS = 1 / (1 - 0.5) = 1 / 0.5 = 2\nThe series converges to 2.",
        "g": 30
    },
    {
        "f": "math_vector_83.png",
        "q": "Find the dot product of vectors A = (3, 4) and B = (1, 2).",
        "s": "A · B = 3*1 + 4*2 = 3 + 8 = 11",
        "a": "A · B = Ax*Bx + Ay*By = 3*1 + 4*2 = 3 + 8 = 11",
        "g": 83
    },
    {
        "f": "math_statistics_56.png",
        "q": "Find the mean and standard deviation of: 2, 4, 4, 4, 5, 5, 7, 9",
        "s": "Mean = (2+4+4+4+5+5+7+9)/8 = 40/8 = 5",
        "a": "Mean = 40/8 = 5\nVariance = avg of squared deviations = 4\nStd dev = √4 = 2",
        "g": 56
    },
    {
        "f": "math_linear_95.png",
        "q": "Solve the system: x + y = 5 and 2x - y = 1",
        "s": "Add equations: 3x = 6, x = 2.\nThen y = 5 - 2 = 3.",
        "a": "Add both equations: 3x = 6 → x = 2\nSubstitute: 2 + y = 5 → y = 3\nSolution: (x=2, y=3)",
        "g": 95
    },
    {
        "f": "math_calculus2_0.png",
        "q": "What is the chain rule in differentiation?",
        "s": "You multiply the derivatives.",
        "a": "If h(x) = f(g(x)) then h'(x) = f'(g(x)) * g'(x)\nExample: d/dx[sin(x^2)] = cos(x^2) * 2x",
        "g": 0
    },

    # ── LITERATURE (8) ────────────────────────────────────────────────────────
    {
        "f": "lit_theme_100.png",
        "q": "What is the central theme of Shakespeare's Hamlet?",
        "s": "The central theme is revenge and its moral consequences.\nHamlet's delay in avenging his father reflects\nhis internal conflict and philosophical doubt.",
        "a": "Central themes include revenge, moral corruption,\nand existential doubt.\nHamlet's hesitation to act reveals a man paralyzed\nby philosophical questioning of life and death.",
        "g": 100
    },
    {
        "f": "lit_character_78.png",
        "q": "Describe the character development of Atticus Finch in To Kill a Mockingbird.",
        "s": "Atticus is a moral, principled lawyer who defends\na Black man despite social pressure. He teaches his\nchildren about empathy and justice.",
        "a": "Atticus Finch represents moral integrity and justice.\nHe defends Tom Robinson despite racial prejudice,\nteaching Scout and Jem to judge people fairly\nand to show empathy for others.",
        "g": 78
    },
    {
        "f": "lit_metaphor_55.png",
        "q": "What is a metaphor? Give an example from literature.",
        "s": "A metaphor compares two things. Like 'life is a journey'.",
        "a": "A metaphor is a direct comparison between two unlike\nthings without using 'like' or 'as'.\nExample: 'All the world's a stage' (Shakespeare)\nCompares life to a theatrical performance.",
        "g": 55
    },
    {
        "f": "lit_essay_85.png",
        "q": "Explain the narrative structure of 1984 by George Orwell.",
        "s": "1984 uses a third-person limited perspective following\nWinston Smith in a totalitarian society. The story\nbuilds tension through surveillance and rebellion.",
        "a": "Third-person limited narration follows Winston Smith\nin a dystopian Oceania under Big Brother.\nThe narrative arc: oppression → rebellion → defeat,\nreflecting Orwell's warning about totalitarianism.",
        "g": 85
    },
    {
        "f": "lit_poetry_40.png",
        "q": "What is iambic pentameter? Identify it in Shakespeare.",
        "s": "It is a type of rhythm in poetry with ten syllables.",
        "a": "Iambic pentameter = 5 iambic feet per line.\nAn iamb is an unstressed then stressed syllable (da-DUM).\nExample: 'Shall I com-PARE thee TO a SUM-mer's DAY'\n10 syllables, alternating stress.",
        "g": 40
    },
    {
        "f": "lit_symbolism_90.png",
        "q": "What does the green light symbolize in The Great Gatsby?",
        "s": "The green light at Daisy's dock symbolizes Gatsby's\ndream and longing. It represents the unattainable\nAmerican Dream and hope for the future.",
        "a": "The green light symbolizes Gatsby's hope and dreams,\nspecifically his desire for Daisy and the American Dream.\nFitzgerald uses it to critique the illusion of the\nAmerican Dream as permanently out of reach.",
        "g": 90
    },
    {
        "f": "lit_narrative_25.png",
        "q": "What is the difference between first-person and third-person omniscient narration?",
        "s": "First person uses 'I'. Third person uses 'he' or 'she'.",
        "a": "First-person: narrator is a character using 'I',\nlimited to their own perspective.\nThird-person omniscient: external narrator knows\nall characters' thoughts and events simultaneously.",
        "g": 25
    },
    {
        "f": "lit_analysis_0.png",
        "q": "What literary device is used in: 'The wind whispered through the trees'?",
        "s": "It is a simile.",
        "a": "This is personification — giving human qualities\n(whispering) to a non-human thing (the wind).\nNot a simile, which requires 'like' or 'as'.",
        "g": 0
    },

    # ── HISTORY (7) ───────────────────────────────────────────────────────────
    {
        "f": "hist_ww2_100.png",
        "q": "What were the main causes of World War II?",
        "s": "WW2 was caused by the rise of Nazism in Germany,\nthe failure of the Treaty of Versailles, economic\ndepression, and appeasement policies.",
        "a": "Main causes: rise of fascism/Nazism, harsh terms of\nVersailles Treaty fueling German resentment,\nglobal economic depression, failed appeasement,\nand aggressive expansion by Germany, Italy, Japan.",
        "g": 100
    },
    {
        "f": "hist_revolution_70.png",
        "q": "What triggered the French Revolution?",
        "s": "Economic crisis and inequality between classes\ncaused the French Revolution.",
        "a": "Triggers: severe financial crisis, food shortages,\nheavy taxation of the poor, Enlightenment ideals,\nand resentment of royal absolutism.\nThe Estates-General deadlock was the breaking point.",
        "g": 70
    },
    {
        "f": "hist_empire_45.png",
        "q": "Why did the Roman Empire fall?",
        "s": "Rome fell because it was attacked by barbarians.",
        "a": "Multi-causal: military overextension, economic decline,\ncorruption and political instability, Germanic invasions,\nand the division into Eastern and Western empires.\nNo single cause; ongoing debate among historians.",
        "g": 45
    },
    {
        "f": "hist_medieval_80.png",
        "q": "What was the feudal system in medieval Europe?",
        "s": "Feudalism was a hierarchy where lords gave land to\nknights and serfs in exchange for military service\nand labor.",
        "a": "Feudalism: hierarchical land-for-service system.\nKing → Lords → Knights → Serfs.\nLords granted fiefs (land) to vassals who pledged\nmilitary service. Serfs worked land for protection.",
        "g": 80
    },
    {
        "f": "hist_ancient_60.png",
        "q": "What were the main contributions of ancient Greece to Western civilization?",
        "s": "Greece contributed democracy, philosophy, and the Olympics.",
        "a": "Democracy, philosophy (Socrates, Plato, Aristotle),\nmathematics, science, theater (tragedy/comedy),\nOlympic games, and architectural styles\nstill influencing Western civilization.",
        "g": 60
    },
    {
        "f": "hist_colonial_35.png",
        "q": "What was the main economic system of European colonialism?",
        "s": "Colonies traded goods with Europe.",
        "a": "Mercantilism: colonies existed to enrich the mother\ncountry by providing raw materials and consuming\nmanufactured goods, maintaining a trade surplus.\nThis justified exploitation of colonial resources.",
        "g": 35
    },
    {
        "f": "hist_cold_war_88.png",
        "q": "What was the Cuban Missile Crisis and how was it resolved?",
        "s": "The Soviet Union placed nuclear missiles in Cuba in\n1962. Kennedy ordered a naval blockade. After tense\nnegotiations, Soviets removed missiles.",
        "a": "1962: USSR placed nuclear missiles in Cuba.\nKennedy imposed naval quarantine.\nResolved: USSR removed missiles; US pledged not to\ninvade Cuba and secretly removed missiles from Turkey.",
        "g": 88
    },

    # ── CHEMISTRY (7) ─────────────────────────────────────────────────────────
    {
        "f": "chem_reaction_100.png",
        "q": "Balance the equation: H2 + O2 → H2O",
        "s": "2H2 + O2 → 2H2O",
        "a": "2H2 + O2 → 2H2O\nLeft: 4H, 2O  |  Right: 4H, 2O  ✓ balanced",
        "g": 100
    },
    {
        "f": "chem_periodic_75.png",
        "q": "What are the properties of noble gases?",
        "s": "Noble gases are in group 18. They are inert and do\nnot react easily because their outer shell is full.",
        "a": "Noble gases (Group 18): full outer electron shell,\nextremely stable and unreactive,\nmonoatomic, low boiling points,\nexamples: He, Ne, Ar, Kr, Xe, Rn.",
        "g": 75
    },
    {
        "f": "chem_bond_50.png",
        "q": "What is the difference between ionic and covalent bonds?",
        "s": "Ionic bonds involve metals and non-metals.\nCovalent bonds involve non-metals.",
        "a": "Ionic: electron transfer between metal and non-metal,\nforming oppositely charged ions (e.g. NaCl).\nCovalent: electron sharing between non-metals\n(e.g. H2O, CO2). Generally lower melting point.",
        "g": 50
    },
    {
        "f": "chem_acid_base_92.png",
        "q": "What is the pH scale and what values indicate acid vs base?",
        "s": "pH 0-14. Below 7 is acidic, above 7 is basic, 7 is neutral.",
        "a": "pH scale 0–14 measures hydrogen ion concentration.\npH < 7: acidic (more H+ ions)\npH = 7: neutral (pure water)\npH > 7: basic/alkaline (more OH- ions)",
        "g": 92
    },
    {
        "f": "chem_organic_65.png",
        "q": "What is the difference between saturated and unsaturated hydrocarbons?",
        "s": "Saturated has no double bonds. Unsaturated has double bonds.",
        "a": "Saturated (alkanes): only single C-C bonds, formula CnH(2n+2).\nUnsaturated: contain double (alkenes) or triple\n(alkynes) bonds. Less hydrogen, more reactive.",
        "g": 65
    },
    {
        "f": "chem_titration_82.png",
        "q": "Explain the process of acid-base titration.",
        "s": "You slowly add a base to an acid until the reaction\nis complete. An indicator shows the endpoint.",
        "a": "Titration: slowly add a solution of known concentration\n(titrant) to the analyte until equivalence point.\nIndicator changes color at endpoint.\nUsed to find unknown concentration: C1V1 = C2V2.",
        "g": 82
    },
    {
        "f": "chem_equation_20.png",
        "q": "What is the molar mass of water (H2O)?",
        "s": "The molar mass of water is 16 g/mol.",
        "a": "H2O: 2 × H (1 g/mol) + 1 × O (16 g/mol)\n= 2 + 16 = 18 g/mol",
        "g": 20
    },

    # ── BIOLOGY (7) ───────────────────────────────────────────────────────────
    {
        "f": "bio_mitosis_100.png",
        "q": "List and describe the stages of mitosis.",
        "s": "Prophase, Metaphase, Anaphase, Telophase.\nProphase: chromosomes condense.\nMetaphase: align at center.\nAnaphase: separate to poles.\nTelophase: nuclear envelopes reform.",
        "a": "PMAT: Prophase (chromatin condenses, spindle forms),\nMetaphase (chromosomes align at plate),\nAnaphase (sister chromatids pulled to poles),\nTelophase (nuclear envelope reforms, cytokinesis).",
        "g": 100
    },
    {
        "f": "bio_dna_80.png",
        "q": "What is the structure of DNA?",
        "s": "DNA is a double helix made of nucleotides.\nEach nucleotide has a sugar, phosphate, and a base.\nBases pair: A-T and G-C.",
        "a": "Double helix: two antiparallel strands of nucleotides.\nNucleotide = deoxyribose sugar + phosphate + nitrogenous base.\nComplementary base pairing: A-T (2 H-bonds), G-C (3 H-bonds).\nDiscovered by Watson and Crick (1953).",
        "g": 80
    },
    {
        "f": "bio_photosynthesis_60.png",
        "q": "Write the equation for photosynthesis.",
        "s": "CO2 + H2O → glucose + O2 using sunlight.",
        "a": "6CO2 + 6H2O + light energy → C6H12O6 + 6O2\nOccurs in chloroplasts.\nLight reactions (thylakoid) + Calvin cycle (stroma).",
        "g": 60
    },
    {
        "f": "bio_genetics_70.png",
        "q": "What is Mendel's law of segregation?",
        "s": "Alleles separate during reproduction so each gamete\ngets one allele.",
        "a": "Each organism has two alleles for each trait.\nDuring gamete formation (meiosis), the two alleles\nsegregate so each gamete carries only one allele.\nOffspring inherit one allele from each parent.",
        "g": 70
    },
    {
        "f": "bio_evolution_42.png",
        "q": "What is natural selection?",
        "s": "Stronger animals survive and reproduce.",
        "a": "Natural selection: organisms with traits better suited\nto their environment survive and reproduce more.\nOver generations, advantageous traits increase\nin frequency — driving evolution (Darwin).",
        "g": 42
    },
    {
        "f": "bio_cell_90.png",
        "q": "What is the difference between prokaryotic and eukaryotic cells?",
        "s": "Prokaryotes have no nucleus. Eukaryotes have a nucleus\nand membrane-bound organelles. Bacteria are prokaryotes.",
        "a": "Prokaryotes: no nucleus, no membrane-bound organelles,\nsmaller (1-10μm), e.g. bacteria, archaea.\nEukaryotes: true nucleus, complex organelles,\nlarger (10-100μm), e.g. animals, plants, fungi.",
        "g": 90
    },
    {
        "f": "bio_protein_30.png",
        "q": "Describe the four levels of protein structure.",
        "s": "Proteins have primary and secondary structure.",
        "a": "Primary: amino acid sequence.\nSecondary: local folding (alpha helix, beta sheet)\nfrom H-bonds.\nTertiary: 3D folding of entire chain.\nQuaternary: multiple polypeptide chains together.",
        "g": 30
    },

    # ── ECONOMICS (7) ─────────────────────────────────────────────────────────
    {
        "f": "econ_supply_100.png",
        "q": "Explain the law of supply and the law of demand.",
        "s": "Law of demand: as price rises, quantity demanded falls.\nLaw of supply: as price rises, quantity supplied rises.\nEquilibrium is where they intersect.",
        "a": "Law of demand: inverse relationship between price\nand quantity demanded (ceteris paribus).\nLaw of supply: positive relationship between price\nand quantity supplied. Equilibrium = market clearing price.",
        "g": 100
    },
    {
        "f": "econ_gdp_75.png",
        "q": "What is GDP and how is it calculated?",
        "s": "GDP is the total value of goods and services produced.\nGDP = C + I + G + (X - M)",
        "a": "GDP: total monetary value of all goods and services\nproduced in a country in a year.\nExpenditure approach: GDP = C + I + G + NX\nC=consumption, I=investment, G=government, NX=net exports.",
        "g": 75
    },
    {
        "f": "econ_inflation_85.png",
        "q": "What causes inflation and how do central banks control it?",
        "s": "Inflation is caused by too much money supply or\ndemand exceeding supply. Central banks raise\ninterest rates to reduce inflation.",
        "a": "Causes: demand-pull (excess demand), cost-push\n(rising production costs), monetary expansion.\nControl: central banks raise interest rates to\ncool borrowing and spending, reducing money supply.",
        "g": 85
    },
    {
        "f": "econ_market_45.png",
        "q": "What is market failure? Give two examples.",
        "s": "Market failure is when the market does not work well.",
        "a": "Market failure: when free market allocates resources\ninefficiently.\nExamples: 1. Externalities (pollution = negative externality),\n2. Public goods (non-excludable, non-rival\n   e.g. national defense).",
        "g": 45
    },
    {
        "f": "econ_fiscal_65.png",
        "q": "What is the difference between fiscal policy and monetary policy?",
        "s": "Fiscal policy involves government spending and taxes.\nMonetary policy involves interest rates and money supply.",
        "a": "Fiscal policy: government adjusts spending and taxes\nto influence the economy (controlled by government).\nMonetary policy: central bank controls money supply\nand interest rates to stabilize prices and growth.",
        "g": 65
    },
    {
        "f": "econ_trade_32.png",
        "q": "What is comparative advantage in international trade?",
        "s": "A country should produce what it makes cheapest.",
        "a": "Comparative advantage: a country should specialize\nin goods it produces at lower opportunity cost\nthan other countries, even if it's not absolutely\nbetter at producing them (Ricardo's theory).",
        "g": 32
    },
    {
        "f": "econ_monopoly_58.png",
        "q": "What is a monopoly and why is it considered problematic?",
        "s": "A monopoly is when one company controls the market.\nIt can charge high prices.",
        "a": "Monopoly: single seller controls an entire market.\nProblematic because: it sets prices above competitive level,\nreduces consumer surplus, causes deadweight loss,\nand reduces innovation incentives.",
        "g": 58
    },

    # ── CYBERSECURITY (5) ─────────────────────────────────────────────────────
    {
        "f": "cyber_sql_injection_100.png",
        "q": "What is a SQL injection attack? How do you prevent it?",
        "s": "SQL injection is when an attacker injects SQL code\ninto an input field. Prevention: use prepared\nstatements and parameterized queries.",
        "a": "SQL injection: attacker inserts malicious SQL into\nan input (e.g. ' OR 1=1 --) to manipulate queries.\nPrevention: parameterized queries/prepared statements,\nORM usage, input validation, least privilege DB user.",
        "g": 100
    },
    {
        "f": "cyber_hash_78.png",
        "q": "What is password hashing and why should you use salt?",
        "s": "Hashing converts a password to a fixed string.\nSalt adds random data to prevent rainbow table attacks.",
        "a": "Hashing: one-way transformation of password using\nalgorithms like bcrypt, Argon2.\nSalt: unique random value added before hashing.\nPrevents rainbow table attacks and ensures\ntwo identical passwords produce different hashes.",
        "g": 78
    },
    {
        "f": "cyber_xss_85.png",
        "q": "What is a Cross-Site Scripting (XSS) attack?",
        "s": "XSS injects malicious scripts into a webpage that\nexecutes in other users' browsers to steal data.",
        "a": "XSS: attacker injects malicious JavaScript into\na trusted website. Victims' browsers execute it.\nTypes: Stored, Reflected, DOM-based.\nPrevention: output encoding, Content-Security-Policy,\nvalidating and sanitizing user input.",
        "g": 85
    },
    {
        "f": "cyber_rsa_55.png",
        "q": "Explain how RSA encryption works at a high level.",
        "s": "RSA uses a public and private key pair.\nYou encrypt with public key, decrypt with private key.",
        "a": "RSA: asymmetric encryption based on difficulty of\nfactoring large prime products.\nKey gen: pick two large primes p,q; compute n=p*q.\nPublic key (n,e); private key (n,d).\nEncrypt: c = m^e mod n. Decrypt: m = c^d mod n.",
        "g": 55
    },
    {
        "f": "cyber_firewall_40.png",
        "q": "What is the difference between a stateful and stateless firewall?",
        "s": "A stateful firewall tracks connections.",
        "a": "Stateless: filters packets based on static rules\n(source/destination IP, port) independently.\nStateful: tracks full connection state (SYN/ACK),\nallowing rules based on connection context.\nStateful is more secure but more resource-intensive.",
        "g": 40
    },

    # ── EXTRA CS (10) ─────────────────────────────────────────────────────────
    {
        "f": "ds_heap_100.png",
        "q": "What is a min-heap? What is the time complexity of insert and extract-min?",
        "s": "A min-heap is a complete binary tree where each node\nis smaller than its children. Insert: O(log n).\nExtract-min: O(log n).",
        "a": "Min-heap: complete binary tree, parent ≤ children.\nRoot is always the minimum element.\nInsert: add at end, bubble up → O(log n).\nExtract-min: remove root, replace with last,\nsift down → O(log n).",
        "g": 100
    },
    {
        "f": "ds_avl_72.png",
        "q": "What is an AVL tree and why is it used?",
        "s": "An AVL tree is a self-balancing BST.\nIt rotates when the height difference is more than 1.",
        "a": "AVL tree: self-balancing BST where |height(left) -\nheight(right)| ≤ 1 for every node.\nRebalances via rotations after insert/delete.\nGuarantees O(log n) search, insert, delete.",
        "g": 72
    },
    {
        "f": "ds_graph_85.png",
        "q": "What is the difference between BFS and DFS? When would you use each?",
        "s": "BFS uses a queue and explores level by level.\nDFS uses a stack and goes deep first.\nBFS for shortest path, DFS for cycle detection.",
        "a": "BFS (queue): level-by-level, good for shortest path\nin unweighted graphs, O(V+E).\nDFS (stack/recursion): explores deep paths,\ngood for cycle detection, topological sort,\nconnected components. O(V+E).",
        "g": 85
    },
    {
        "f": "oop_polymorphism_90.png",
        "q": "What is polymorphism in OOP? Give a Java example.",
        "s": "Polymorphism means one interface, multiple forms.\nExample: Animal a = new Dog(); a.speak();\ncalls Dog's speak method.",
        "a": "Polymorphism: same interface used for different types.\nRuntime (dynamic): method overriding via inheritance.\nCompile-time (static): method overloading.\nExample: Animal a = new Dog();\na.makeSound(); // calls Dog's overridden makeSound()",
        "g": 90
    },
    {
        "f": "oop_inheritance_60.png",
        "q": "What is the difference between inheritance and composition in OOP?",
        "s": "Inheritance means a class extends another class.\nComposition means a class has another class as a field.",
        "a": "Inheritance (is-a): subclass extends superclass,\ninherits methods/fields. Tight coupling.\nComposition (has-a): class contains instance of\nanother class as a field. Preferred for flexibility.\n'Favor composition over inheritance' (GoF).",
        "g": 60
    },
    {
        "f": "db_normalization_95.png",
        "q": "What is database normalization? Explain 1NF, 2NF, 3NF.",
        "s": "Normalization removes data redundancy.\n1NF: atomic values, no repeating groups.\n2NF: no partial dependency on composite key.\n3NF: no transitive dependency.",
        "a": "1NF: atomic column values, unique rows.\n2NF: 1NF + all non-key columns depend on entire PK.\n3NF: 2NF + no transitive dependencies\n(non-key column depends only on PK, not other non-keys).",
        "g": 95
    },
    {
        "f": "db_index_68.png",
        "q": "What is a database index and what are its trade-offs?",
        "s": "An index speeds up searches but uses extra space\nand slows down inserts and updates.",
        "a": "Index: data structure (B-tree or hash) that speeds\nup SELECT queries on indexed columns.\nTrade-offs: faster reads O(log n),\nslower writes (INSERT/UPDATE/DELETE must update index),\nextra storage overhead.",
        "g": 68
    },
    {
        "f": "sys_cache_80.png",
        "q": "What is cache memory and what is the principle of locality?",
        "s": "Cache is fast memory between CPU and RAM.\nLocality means recently used data is likely\nto be used again.",
        "a": "Cache: small, fast memory closer to CPU than RAM.\nTemporal locality: recently accessed data likely\nreused soon.\nSpatial locality: data near recently accessed\naddresses likely accessed next.\nCache exploits both to reduce memory latency.",
        "g": 80
    },
    {
        "f": "sys_virtual_memory_48.png",
        "q": "What is virtual memory and why is it used?",
        "s": "Virtual memory lets programs use more memory than\nphysically available.",
        "a": "Virtual memory: OS abstraction giving each process\nits own address space, backed by RAM and disk (swap).\nBenefits: isolation between processes, allows programs\nlarger than RAM, simplifies memory management.\nImplemented via paging/segmentation.",
        "g": 48
    },
    {
        "f": "cloud_docker_76.png",
        "q": "What is Docker and how does it differ from a virtual machine?",
        "s": "Docker uses containers that are lighter than VMs.\nContainers share the OS kernel, VMs have their own OS.",
        "a": "Docker: containerization platform packaging app + deps.\nContainers share host OS kernel → lightweight, fast.\nVMs emulate full hardware + run separate OS → heavier.\nDocker: seconds to start, MB in size.\nVM: minutes to start, GB in size.",
        "g": 76
    },
]


def create_image_from_text(text, filename, folder):
    W, H = 800, 600
    img = Image.new('RGB', (W, H), color='white')
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except:
        font = ImageFont.load_default()

    for i in range(50, H, 50):
        d.line([(0, i), (W, i)], fill=(200, 220, 255), width=2)
    d.line([(60, 0), (60, H)], fill=(255, 200, 200), width=2)

    lines = text.split('\n')
    y_text = 65
    for line in lines:
        d.text((70, y_text), line, fill=(0, 0, 0), font=font)
        y_text += 35

    path = os.path.join(folder, filename)
    img.save(path)


print(f"Generating {len(new_data)} new exam entries...")

# Load existing CSV if it exists
if os.path.exists(CSV_FILE):
    existing = pd.read_csv(CSV_FILE)
    existing_files = set(existing['filename'].tolist())
else:
    existing = pd.DataFrame(columns=['filename', 'grade'])
    existing_files = set()

new_rows = []
for item in new_data:
    if item['f'] in existing_files:
        print(f"  SKIP (already exists): {item['f']}")
        continue
    create_image_from_text(item['q'], item['f'], QUESTIONS_DIR)
    create_image_from_text(item['s'], item['f'], SOLUTIONS_DIR)
    create_image_from_text(item['a'], item['f'], ANSWERS_DIR)
    new_rows.append({'filename': item['f'], 'grade': item['g']})
    print(f"  Created: {item['f']} (grade: {item['g']})")

if new_rows:
    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    combined.to_csv(CSV_FILE, index=False)
    print(f"\nDone. Added {len(new_rows)} new entries.")
    print(f"Total in train.csv: {len(combined)}")
else:
    print("No new entries added.")
