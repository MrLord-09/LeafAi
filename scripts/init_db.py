import json, sqlite3, os
BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH=os.path.join(BASE,'backend','official_cures.db')
CATALOG_PATH=os.path.join(BASE,'data','disease_catalog.json')

PLANTS={
'Apple':['healthy','Apple_scab','Black_rot','Cedar_apple_rust'],
'Blueberry':['healthy'],'Cherry':['healthy','Powdery_mildew'],'Corn':['healthy','Cercospora_leaf_spot_Gray_leaf_spot','Common_rust','Northern_Leaf_Blight'],
'Grape':['healthy','Black_rot','Esca_(Black_MeASLES)','Leaf_blight_(Isariopsis_Leaf_Spot)'],'Peach':['healthy','Bacterial_spot'],'Pepper_bell':['healthy','Bacterial_spot'],
'Potato':['healthy','Early_blight','Late_blight'],'Raspberry':['healthy'],'Soybean':['healthy'],'Squash':['Powdery_mildew'],'Strawberry':['healthy','Leaf_scorch'],
'Tomato':['healthy','Bacterial_spot','Early_blight','Late_blight','Leaf_Mold','Septoria_leaf_spot','Spider_mites_Two-spotted_spider_mite','Target_Spot','Tomato_Yellow_Leaf_Curl_Virus','Tomato_mosaic_virus'],
'Mango':['healthy','Anthracnose','Powdery_Mildew','Bacterial_Canker','Sooty_Mold'],
'Rice':['healthy','Blast','Brown_Spot','Bacterial_Leaf_Blight','Sheath_Blight'],
'Wheat':['healthy','Leaf_Rust','Stem_Rust','Stripe_Rust','Powdery_Mildew'],
'Banana':['healthy','Black_Sigatoka','Panama_Disease','Bunchy_Top_Virus'],
'Cotton':['healthy','Bacterial_Blight','Fusarium_Wilt','Leaf_Curl_Virus'],
'Papaya':['healthy','Ring_Spot_Virus','Powdery_Mildew','Anthracnose'],
'Guava':['healthy','Wilt','Anthracnose','Algal_Leaf_Spot'],
'Onion':['healthy','Purple_Blotch','Downy_Mildew','Stemphylium_Blight'],
'Cabbage':['healthy','Black_Rot','Downy_Mildew','Club_Root'],
'Chilli':['healthy','Leaf_Curl_Virus','Anthracnose','Bacterial_Spot'],
'Brinjal':['healthy','Little_Leaf_Disease','Phomopsis_Blight','Verticillium_Wilt'],
'Groundnut':['healthy','Tikka_Disease','Rust','Collar_Rot'],
'Sunflower':['healthy','Alternaria_Blight','Rust','Downy_Mildew'],
'Tea':['healthy','Blister_Blight','Red_Rust','Grey_Blight']}

# Existing PlantVillage labels are kept exactly where practical.
PV=['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy','Blueberry___healthy','Cherry___Powdery_mildew','Cherry___healthy','Corn___Cercospora_leaf_spot Gray_leaf_spot','Corn___Common_rust','Corn___Northern_Leaf_Blight','Corn___healthy','Grape___Black_rot','Grape___Esca_(Black_MeASLES)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy','Peach___Bacterial_spot','Peach___healthy','Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy','Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy']

with open(CATALOG_PATH,encoding='utf-8') as f: catalog=json.load(f)
labels_path=os.path.join(BASE,'backend','class_names.json')
with open(labels_path,encoding='utf-8') as f: trained_classes=json.load(f)
conn=sqlite3.connect(DB_PATH); cur=conn.cursor()
cur.execute('''CREATE TABLE IF NOT EXISTS plant_diseases(id INTEGER PRIMARY KEY AUTOINCREMENT,class_name TEXT UNIQUE NOT NULL,disease_name TEXT NOT NULL,symptoms TEXT NOT NULL,organic_control TEXT NOT NULL,chemical_control TEXT NOT NULL,source TEXT NOT NULL)''')
cur.execute('CREATE INDEX IF NOT EXISTS idx_class_name ON plant_diseases(class_name)')
rows=[]
for cls, rec in catalog.items(): rows.append((cls,rec['disease_name'],rec['symptoms'],rec['organic_control'],rec['chemical_control'],rec['source']))
# Ensure every class in the currently trained model has a complete guide record.
existing_classes={row[0] for row in rows}
for cls in trained_classes:
    if cls in existing_classes:
        continue
    rec=catalog.get(cls,{})
    disease_name=rec.get('disease_name',cls.replace('___',' - ').replace('_',' '))
    rows.append((
        cls,
        disease_name,
        rec.get('symptoms','Visual classification only; confirm the diagnosis with a local agricultural expert.'),
        rec.get('organic_control','Use sanitation, crop-specific cultural practices and integrated pest management after confirming the cause.'),
        rec.get('chemical_control','Do not choose a pesticide from this label alone. Confirm the target and use only a product currently registered in India for this crop and disease, strictly following its label.'),
        rec.get('source','Local agricultural extension; confirm current Indian label')
    ))
# Healthy entries for every crop.
for crop,diseases in PLANTS.items():
    cls=f'{crop}___healthy'
    if cls not in catalog:
        rows.append((cls,f'Healthy {crop}',f'No characteristic disease symptoms are visible in the submitted image of the {crop} plant.',f'Maintain balanced irrigation and nutrition, sanitation, airflow and regular scouting.',f'No disease treatment is indicated for a healthy plant. Use pesticides only when a verified pest/disease is present.', 'General crop health/IPM guidance'))
# Generic records for PlantVillage classes that are not in the custom catalog.
for cls in PV:
    if not any(r[0]==cls for r in rows):
        name=cls.split('___',1)[1].replace('_',' ').replace('  ',' ')
        rows.append((cls,name,f'Visual symptoms associated with {name}. Confirm with crop-specific scouting before treatment.', 'Remove affected material, improve airflow, sanitation and use integrated pest management.', 'Use only a product currently registered/labeled for the crop and disease in your jurisdiction; follow the label and pre-harvest interval.', 'Plant disease IPM guidance; verify current local label'))
cur.executemany('INSERT OR IGNORE INTO plant_diseases(class_name,disease_name,symptoms,organic_control,chemical_control,source) VALUES (?,?,?,?,?,?)',rows)
conn.commit(); print('Seeded',cur.execute('select count(*) from plant_diseases').fetchone()[0],'records into',DB_PATH); conn.close()
