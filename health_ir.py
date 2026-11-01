import math, re
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer

# ---------- 1. PREPROCESSING ----------
STOP = set("a an the and or of to in on for with is are was were be been it its this that these those at by from as can not no do does what how when which who into than then so such may also often more most other their there they them have has had will".split())
stem = PorterStemmer().stem

def preprocess(text):
    tokens = re.findall(r"[a-z0-9]+", text.lower())      # tokenize
    return [stem(t) for t in tokens if t not in STOP]     # stopwords + stemming

# ---------- 2. DATASET (id, title, category, text) ----------
RAW = [
(1,"Diabetes","Metabolic","Diabetes is a long-term condition in which the body cannot control glucose in the blood. Symptoms include increased thirst, frequent urination, tiredness, blurred vision and slow-healing wounds. Management includes diet, exercise, glucose monitoring, tablets or insulin."),
(2,"Hypertension","Cardiovascular","Hypertension means blood pressure stays higher than normal. It often has no symptoms but can cause headache and dizziness. Reducing salt, exercise, healthy weight and prescribed tablets control it and lower the risk of stroke and heart disease."),
(3,"Dengue Fever","Infectious","Dengue is a viral infection spread by Aedes mosquitoes. It causes sudden high fever, severe headache, pain behind the eyes, joint and muscle pain and a skin rash. Treatment is rest, fluids and paracetamol. Prevent it by removing stagnant water where mosquitoes breed."),
(4,"Asthma","Respiratory","Asthma narrows and inflames the airways. Symptoms are wheezing, shortness of breath, chest tightness and cough, often worse at night. Triggers include dust, pollen, smoke and cold air. Inhalers are used for relief and long-term control."),
(5,"Anemia","Blood","Anemia means too few healthy red blood cells. Symptoms include fatigue, pale skin, weakness and dizziness. Iron deficiency is the commonest cause. Iron-rich foods like leafy greens and lentils, and supplements advised by a doctor, treat it."),
(6,"Migraine","Neurological","Migraine is a recurring headache with throbbing pain on one side, nausea and sensitivity to light and sound. Triggers include stress, irregular sleep and skipped meals. Rest in a dark quiet room and pain relief help."),
(7,"First Aid for Burns","First Aid","For a minor burn, cool the area under running water for twenty minutes. Remove rings near the burn and cover loosely with a clean dressing. Do not apply butter or toothpaste and do not burst blisters. Large or deep burns need emergency care."),
(8,"Malaria","Infectious","Malaria is caused by parasites carried by Anopheles mosquitoes. Symptoms are fever with chills and sweating, headache, vomiting and body aches. A blood test confirms it and antimalarial medicines treat it. Mosquito nets and repellents prevent it."),
(9,"Tuberculosis","Respiratory","Tuberculosis is a bacterial infection of the lungs spread through coughing. Symptoms include a cough lasting more than three weeks, weight loss, night sweats, fever and chest pain. It is curable with six months of antibiotics."),
(10,"Common Cold","Respiratory","The common cold is a mild viral infection causing runny nose, sneezing, sore throat and mild cough. Antibiotics do not work. Rest, warm fluids and handwashing help."),
(11,"Heart Attack","Cardiovascular","A heart attack happens when blood flow to the heart muscle is blocked. Warning signs are chest pain or pressure spreading to the arm or jaw, sweating, breathlessness and nausea. Call emergency services immediately."),
(12,"Stroke","Neurological","A stroke occurs when blood supply to part of the brain is cut off. Signs include face drooping, arm weakness, slurred speech and confusion. Call emergency services at once. Controlling blood pressure lowers the risk."),
(13,"Gastroenteritis and Diarrhoea","Digestive","Gastroenteritis causes diarrhoea, vomiting, stomach cramps and fever from infection or contaminated food. The main treatment is oral rehydration solution to replace lost fluids. See a doctor if there is blood in stool or severe dehydration."),
(14,"Urinary Tract Infection","Urinary","A urinary tract infection is a bacterial infection of the bladder. Symptoms include a burning sensation when passing urine, frequent urge to urinate and cloudy urine. Antibiotics treat it; drinking water helps prevent it."),
(15,"Depression","Mental Health","Depression causes persistent sadness, hopelessness, loss of interest, tiredness and changes in sleep and appetite. Therapy, medicines, exercise and support help recovery. Thoughts of self-harm need immediate help."),
(16,"Jaundice","Liver","Jaundice is yellowing of the skin and eyes caused by excess bilirubin, usually from liver problems such as hepatitis. It may come with dark urine, itching and fatigue. Rest, good nutrition and avoiding alcohol help recovery."),
(17,"Influenza","Respiratory","Influenza, or flu, is a contagious respiratory illness. It starts suddenly with high fever, body aches, headache, dry cough, sore throat and extreme tiredness. It can cause pneumonia in older adults and young children. Rest, fluids and fever medicines help, and yearly vaccination is the best protection."),
(18,"Typhoid Fever","Infectious","Typhoid fever is caused by Salmonella Typhi bacteria and spreads through contaminated food and water. Symptoms include a prolonged high fever, weakness, stomach pain, headache, constipation or diarrhoea. It is diagnosed by a blood culture and treated with antibiotics. Boiled water, handwashing and vaccination prevent it."),
(19,"Dehydration","Digestive","Dehydration occurs when the body loses more fluid than it takes in through vomiting, diarrhoea, fever or heavy sweating. Signs include dry mouth, thirst, dark urine, dizziness and in children sunken eyes. Oral rehydration solution made of clean water, salt and sugar restores fluid balance. Severe cases need intravenous fluids."),
(20,"Obesity","Lifestyle","Obesity is excess body fat that harms health, usually defined as a BMI of thirty or more. It raises the risk of diabetes, hypertension, heart disease and joint problems. Causes include unhealthy diet, inactivity and genetics. Balanced meals, portion control and regular exercise help with gradual weight loss."),
(21,"Anxiety Disorder","Mental Health","Anxiety disorders involve excessive worry or fear that is hard to control. Symptoms include restlessness, rapid heartbeat, sweating, trembling, trouble sleeping and panic attacks. Therapy, breathing exercises, physical activity and sometimes medicines help. Limiting caffeine reduces symptoms."),
(22,"Hypothyroidism","Endocrine","Hypothyroidism occurs when the thyroid gland in the neck makes too little thyroid hormone. Symptoms include tiredness, weight gain, feeling cold, dry skin, constipation and hair thinning. A blood test for TSH confirms it. Daily levothyroxine tablets are the treatment."),
(23,"Kidney Stones","Urinary","Kidney stones are hard deposits of minerals that form inside the kidneys. A stone moving down the ureter causes severe cramping pain in the side and back, nausea, blood in urine and painful urination. Drinking plenty of water prevents them. Small stones pass on their own, while large stones may need lithotripsy."),
(24,"Allergies","Immunology","An allergy is an exaggerated immune reaction to substances such as pollen, dust mites, certain foods or pet dander. Symptoms include sneezing, itchy watery eyes, runny nose, skin rash and hives. Severe reactions called anaphylaxis cause throat swelling and need emergency care. Avoiding triggers and antihistamines are the main approaches."),
(25,"Chickenpox","Infectious","Chickenpox is a highly contagious viral illness. It causes an itchy rash of red spots that become fluid-filled blisters and then scabs, with fever and tiredness. It spreads through coughing and contact with blisters. Calamine lotion and paracetamol relieve discomfort, and a vaccine prevents it."),
(26,"COVID-19","Infectious","COVID-19 is a respiratory illness caused by the SARS-CoV-2 virus. Symptoms include fever, dry cough, sore throat, tiredness, loss of taste or smell and breathing difficulty in severe cases. It spreads through droplets and close contact. Most mild cases recover with rest and fluids, and vaccination reduces severity."),
(27,"Vitamin D Deficiency","Nutrition","Vitamin D helps the body absorb calcium for strong bones. Deficiency causes bone pain, muscle weakness, fatigue and rickets in children. It is common in people with little sunlight exposure. Sunlight, fortified milk, eggs, fatty fish and supplements advised after a blood test correct it."),
(28,"Back Pain","Musculoskeletal","Back pain is common and is often caused by muscle strain, poor posture, lifting heavy objects or long hours of sitting. Most cases improve in a few weeks with staying active, gentle stretching and heat packs. See a doctor if pain comes with leg weakness, numbness or loss of bladder control."),
(29,"Conjunctivitis","Eye","Conjunctivitis, or pink eye, is inflammation of the thin layer covering the white of the eye. It causes redness, itching, watering and sticky discharge. It may be viral, bacterial or allergic. Avoid touching the eyes, do not share towels and consult a doctor for eye drops."),
(30,"Insomnia","Mental Health","Insomnia is difficulty falling asleep, staying asleep or waking too early, leading to daytime tiredness and poor concentration. Fixed sleep timings, avoiding caffeine and screens at night and a dark cool bedroom help. Persistent insomnia needs medical review."),
(31,"Pneumonia","Respiratory","Pneumonia is an infection that inflames the air sacs of the lungs. Symptoms include cough with phlegm, high fever, chills, chest pain and difficulty breathing. Bacterial pneumonia is treated with antibiotics, along with rest and fluids. Older adults, children and smokers are at higher risk."),
(32,"Appendicitis","Digestive","Appendicitis is inflammation of the appendix. It starts as pain around the navel that moves to the lower right abdomen, with nausea, vomiting, fever and loss of appetite. The pain worsens with movement or coughing. It is an emergency and usually treated by surgery."),
(33,"Epilepsy","Neurological","Epilepsy is a brain disorder that causes repeated seizures due to abnormal electrical activity. A seizure may cause shaking, loss of consciousness, staring or confusion. During a seizure, keep the person safe, place them on their side and do not put anything in the mouth. Anti-seizure medicines control it."),
(34,"Arthritis","Musculoskeletal","Arthritis is inflammation of the joints causing pain, swelling and stiffness, especially in the morning. Osteoarthritis comes from wear of the joints, while rheumatoid arthritis is an autoimmune disease. Exercise, weight control, pain relief medicines and physiotherapy help."),
(35,"Acne","Skin","Acne causes pimples, blackheads and oily skin, mostly on the face, when hair follicles become blocked with oil and dead skin. Washing the face twice daily, avoiding squeezing pimples and using creams prescribed by a skin specialist help. Severe acne needs medical treatment to prevent scars."),
(36,"Measles","Infectious","Measles is a highly contagious viral disease. It begins with fever, cough, runny nose and red eyes, followed by a red rash that spreads over the body. It can lead to pneumonia and brain complications in children. There is no specific cure, and two doses of the measles vaccine prevent it."),
]
DOCS = [dict(id=a, title=b, category=c, text=d) for a, b, c, d in RAW]

# Simple query-expansion dictionary (layman term -> medical terms)
SYN = {"sugar": "diabetes glucose", "bp": "blood pressure hypertension",
       "loose motions": "diarrhoea", "loose motion": "diarrhoea",
       "fits": "seizures epilepsy", "flu": "influenza"}

# ---------- 3. SEARCH ENGINE ----------
class SearchEngine:
    def __init__(self, docs):
        self.docs, self.N = docs, len(docs)
        self.index = defaultdict(dict)          # inverted index: term -> {doc_no: tf}
        self.dl = []                            # document lengths
        for i, d in enumerate(docs):
            toks = preprocess(d["title"] + " " + d["text"])
            self.dl.append(len(toks))
            for t, tf in Counter(toks).items():
                self.index[t][i] = tf
        self.avgdl = sum(self.dl) / self.N
        self.idf = {t: math.log(1 + self.N / len(p)) for t, p in self.index.items()}
        n = [0.0] * self.N                      # document vector lengths (for cosine)
        for t, p in self.index.items():
            for i, tf in p.items():
                n[i] += ((1 + math.log(tf)) * self.idf[t]) ** 2
        self.norm = [math.sqrt(x) for x in n]

    def query_terms(self, q, expand=False):
        terms = preprocess(q)
        if expand:
            for key, extra in SYN.items():
                if re.search(r"\b" + key + r"\b", q.lower()):
                    terms += preprocess(extra)
        return terms

    def search(self, q, model="tfidf", expand=False, k=5):
        qt, sc = Counter(self.query_terms(q, expand)), defaultdict(float)
        if model == "tfidf":                    # Vector Space Model: TF-IDF + cosine
            qn = 0.0
            for t, f in qt.items():
                if t not in self.index: continue
                qw = (1 + math.log(f)) * self.idf[t]
                qn += qw * qw
                for i, tf in self.index[t].items():
                    sc[i] += qw * (1 + math.log(tf)) * self.idf[t]
            for i in sc:
                sc[i] /= (self.norm[i] * math.sqrt(qn))
        else:                                   # Probabilistic model: BM25
            k1, b = 1.5, 0.75
            for t in qt:
                if t not in self.index: continue
                df = len(self.index[t])
                idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
                for i, tf in self.index[t].items():
                    sc[i] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * self.dl[i] / self.avgdl))
        ranked = sorted(sc.items(), key=lambda x: -x[1])[:k]
        return [(self.docs[i], s) for i, s in ranked]

# ---------- 4. EVALUATION (run: python health_ir.py) ----------
TESTS = [  # (query, set of relevant doc ids)
 ("high blood sugar symptoms",{1}), ("fever joint pain rash mosquito",{3,8}),
 ("child wheezing at night",{4}), ("sugar problem",{1}), ("high BP treatment",{2}),
 ("chest pain heart attack warning signs",{11}), ("what to do for a burn",{7}),
 ("cough for weeks night sweats weight loss",{9}), ("loose motions",{13}),
 ("feeling sad and hopeless",{15}), ("face drooping slurred speech",{12}),
 ("burning while passing urine",{14}), ("yellow eyes and skin",{16}),
 ("trouble sleeping at night",{30}), ("pain in lower back after lifting",{28}),
 ("itchy watery red eyes",{24,29}), ("severe pain in right lower abdomen",{32}),
 ("sudden seizures and loss of consciousness",{33}), ("fits attack",{33}),
 ("joint pain and stiffness in morning",{34}), ("pimples on face",{35}),
 ("kidney stone pain",{23}), ("how to treat dehydration",{19,13}),
 ("tired feeling cold weight gain",{22}),
]

if __name__ == "__main__":
    eng = SearchEngine(DOCS)
    for name, model, exp in [("TF-IDF","tfidf",False), ("TF-IDF+syn","tfidf",True),
                             ("BM25","bm25",False), ("BM25+syn","bm25",True)]:
        P = R = M = 0
        print(f"\n=== {name} ===")
        for q, rel in TESTS:
            res = eng.search(q, model, exp, 5)
            ids = [d["id"] for d, _ in res]
            hits = len(set(ids[:3]) & rel)
            P += hits / 3; R += hits / len(rel)
            M += next((1 / (r + 1) for r, i in enumerate(ids) if i in rel), 0)
            print(f"{q:42} top3={ids[:3]}  relevant={sorted(rel)}")
        n = len(TESTS)
        print(f"Precision@3={P/n:.2f}  Recall@3={R/n:.2f}  MRR={M/n:.2f}")