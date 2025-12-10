import streamlit as st
from streamlit_option_menu import option_menu
import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ----#

# --- Info for every class in PlantVillage (39 classes) ---
disease_info = {
    "Apple___Apple_scab": {
        "title": "Apple – Apple scab",
        "description": "A common fungal disease (Venturia inaequalis) affecting leaves and fruit, causing scabby lesions and yield loss.",
        "symptoms": [
            "Olive-green to brown velvety spots on leaves",
            "Distorted, cracked, scabby lesions on fruits",
            "Premature leaf drop in severe cases"
        ],
        "prevention": [
            "Remove and destroy fallen leaves and infected fruit",
            "Prune trees to improve air circulation",
            "Use resistant varieties where possible",
            "Apply fungicides preventively during wet spring weather"
        ],
    },
    "Apple___Black_rot": {
        "title": "Apple – Black rot",
        "description": "A fungal disease that infects leaves, fruit and branches, leading to fruit rot and cankers.",
        "symptoms": [
            "Purple spots on leaves that turn brown with a dark border",
            "Black, concentric ring patterns on rotting fruit",
            "Cankers on branches and trunk with sunken, dark areas"
        ],
        "prevention": [
            "Prune and remove dead or cankered wood",
            "Destroy mummified fruit left on the tree or ground",
            "Maintain good pruning to open the canopy",
            "Apply recommended fungicides where pressure is high"
        ],
    },
    "Apple___Cedar_apple_rust": {
        "title": "Apple – Cedar apple rust",
        "description": "A fungal disease that alternates between apple and juniper/cedar hosts.",
        "symptoms": [
            "Bright yellow-orange spots on upper leaf surfaces",
            "Tube-like projections on underside of leaves",
            "Yellow spots on fruit that may crack"
        ],
        "prevention": [
            "Avoid planting apples near cedar/juniper trees when possible",
            "Remove nearby alternate hosts or prune galls on junipers",
            "Use resistant apple cultivars",
            "Apply fungicides in spring if disease history is severe"
        ],
    },
    "Apple___healthy": {
        "title": "Apple – Healthy leaf",
        "description": "Leaf shows no visible signs of disease; normal green color and shape.",
        "symptoms": [
            "Uniform green color without spots or lesions",
            "No distortion, curling or premature yellowing"
        ],
        "prevention": [
            "Maintain regular pruning and sanitation",
            "Avoid overhead irrigation in the evening",
            "Monitor regularly so problems are caught early"
        ],
    },
    "Blueberry___healthy": {
        "title": "Blueberry – Healthy leaf",
        "description": "Healthy blueberry foliage with no visible disease or nutrient problems.",
        "symptoms": [
            "Even green color, no spots, blight or necrosis"
        ],
        "prevention": [
            "Maintain acidic, well-drained soil",
            "Avoid waterlogging and standing water",
            "Scout regularly for insects and leaf spots"
        ],
    },
    "Cherry___Powdery_mildew": {
        "title": "Cherry – Powdery mildew",
        "description": "A fungal disease that coats leaves with a white powdery growth.",
        "symptoms": [
            "White, powdery fungal growth on upper leaf surfaces",
            "Distorted or curled leaves",
            "Premature leaf drop in serious infections"
        ],
        "prevention": [
            "Prune to improve air movement through canopy",
            "Avoid excessive nitrogen which encourages lush, susceptible growth",
            "Apply sulfur or other labeled fungicides at early symptom appearance"
        ],
    },
    "Cherry___healthy": {
        "title": "Cherry – Healthy leaf",
        "description": "Normal cherry foliage without disease.",
        "symptoms": [
            "Glossy green leaves, intact margins, no white powder or spots"
        ],
        "prevention": [
            "Good airflow through pruning",
            "Balanced fertilization and irrigation",
            "Regular scouting for pests and diseases"
        ],
    },
    "Corn___Cercospora_leaf_spot_Gray_leaf_spot": {
        "title": "Corn – Cercospora leaf spot (Gray leaf spot)",
        "description": "A foliar fungal disease that reduces photosynthetic area and grain yield.",
        "symptoms": [
            "Narrow, rectangular gray-tan lesions between leaf veins",
            "Lesions may merge to blight large portions of leaves"
        ],
        "prevention": [
            "Rotate crops away from corn for at least 1–2 years",
            "Reduce surface residue by tillage where appropriate",
            "Plant resistant hybrids when available",
            "Use fungicides in high-risk situations"
        ],
    },
    "Corn___Common_rust": {
        "title": "Corn – Common rust",
        "description": "A fungal disease causing rust-colored pustules on leaves.",
        "symptoms": [
            "Small, cinnamon-brown to dark brown pustules on both leaf surfaces",
            "Pustules may darken and merge as leaves age"
        ],
        "prevention": [
            "Select rust-resistant hybrids",
            "Plant early to avoid heavy late-season infections",
            "Use fungicides when disease is severe and yield potential is high"
        ],
    },
    "Corn___Northern_Leaf_Blight": {
        "title": "Corn – Northern leaf blight",
        "description": "A leaf disease that produces cigar-shaped lesions and can lower yield.",
        "symptoms": [
            "Long, elliptical gray-green lesions on lower leaves",
            "Lesions turn tan and may coalesce"
        ],
        "prevention": [
            "Plant resistant hybrids",
            "Rotate crops to reduce inoculum",
            "Apply fungicides if disease appears early and conditions are favorable"
        ],
    },
    "Corn___healthy": {
        "title": "Corn – Healthy leaf",
        "description": "Corn leaf with normal coloration and no visible infection.",
        "symptoms": [
            "Uniform green leaves with no rust or blight lesions"
        ],
        "prevention": [
            "Use recommended fertilization and irrigation",
            "Scout regularly for early disease signs",
            "Maintain crop rotation to reduce disease pressure"
        ],
    },
    "Grape___Black_rot": {
        "title": "Grape – Black rot",
        "description": "A destructive fungal disease affecting leaves and berries.",
        "symptoms": [
            "Small brown leaf spots with dark margins and black fruiting bodies",
            "Shriveled, black 'mummified' berries"
        ],
        "prevention": [
            "Remove mummified fruit and infected leaves",
            "Prune canopy to improve air circulation",
            "Apply fungicides beginning at early shoot growth"
        ],
    },
    "Grape___Esca_(Black_Measles)": {
        "title": "Grape – Esca (Black measles)",
        "description": "A trunk disease complex that gradually weakens vines.",
        "symptoms": [
            "Tiger-stripe patterns of yellow and brown on leaves",
            "Dark streaks inside trunk and arms",
            "Berries with dark sunken spots and cracking"
        ],
        "prevention": [
            "Avoid large pruning wounds; prune in dry weather",
            "Remove and destroy heavily infected vines",
            "Use clean planting material from certified nurseries"
        ],
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "title": "Grape – Leaf blight (Isariopsis leaf spot)",
        "description": "A foliar disease causing spots and premature defoliation.",
        "symptoms": [
            "Irregular brown lesions on leaves",
            "Yellow halos and early leaf drop"
        ],
        "prevention": [
            "Prune to reduce canopy density",
            "Destroy fallen leaves after leaf fall",
            "Apply protective fungicides as needed"
        ],
    },
    "Grape___healthy": {
        "title": "Grape – Healthy leaf",
        "description": "Normal grapevine foliage.",
        "symptoms": [
            "Even green color, no spotting or yellow tiger stripes"
        ],
        "prevention": [
            "Balanced fertilization and irrigation",
            "Good canopy management",
            "Routine monitoring for mildew and rot"
        ],
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "title": "Orange – Huanglongbing (Citrus greening)",
        "description": "A serious bacterial disease spread by psyllid insects, causing tree decline.",
        "symptoms": [
            "Blotchy yellow leaf mottling (not symmetrical across midrib)",
            "Twig dieback and small, misshapen fruit with bitter taste"
        ],
        "prevention": [
            "Control psyllid vectors using integrated pest management",
            "Use certified disease-free nursery trees",
            "Remove and destroy infected trees where required by local regulations"
        ],
    },
    "Peach___Bacterial_spot": {
        "title": "Peach – Bacterial spot",
        "description": "A bacterial disease causing spots on leaves and fruit.",
        "symptoms": [
            "Small dark leaf spots that may drop out, giving a 'shot-hole' appearance",
            "Sunken, pitted spots on fruit"
        ],
        "prevention": [
            "Plant tolerant varieties",
            "Avoid excessive nitrogen and overhead irrigation",
            "Apply copper sprays when recommended in your region"
        ],
    },
    "Peach___healthy": {
        "title": "Peach – Healthy leaf",
        "description": "Healthy peach foliage.",
        "symptoms": [
            "Smooth, uniform green leaves without shot holes or spots"
        ],
        "prevention": [
            "Regular pruning for airflow",
            "Monitor for insects that can open wounds to infection"
        ],
    },
    "Pepper,_bell___Bacterial_spot": {
        "title": "Bell pepper – Bacterial spot",
        "description": "A bacterial disease of pepper leaves and fruit.",
        "symptoms": [
            "Water-soaked leaf spots that turn brown with yellow halos",
            "Raised, corky spots on fruit"
        ],
        "prevention": [
            "Use disease-free seed or transplants",
            "Avoid working in fields when foliage is wet",
            "Rotate away from solanaceous crops",
            "Apply copper-based bactericides at early stages"
        ],
    },
    "Pepper,_bell___healthy": {
        "title": "Bell pepper – Healthy leaf",
        "description": "Leaf with no visible disease.",
        "symptoms": [
            "Uniform green surface, no water-soaked lesions or yellow halos"
        ],
        "prevention": [
            "Good spacing and airflow",
            "Avoid overhead irrigation late in the day"
        ],
    },
    "Potato___Early_blight": {
        "title": "Potato – Early blight",
        "description": "A fungal disease caused by Alternaria solani, common in older foliage.",
        "symptoms": [
            "Brown spots with concentric rings giving a 'target' pattern",
            "Yellow tissue surrounding lesions",
            "Defoliation starting from older leaves"
        ],
        "prevention": [
            "Rotate crops; avoid planting potatoes after other solanaceous crops",
            "Maintain adequate fertility, especially nitrogen",
            "Apply fungicides protectively when conditions are favorable"
        ],
    },
    "Potato___Late_blight": {
        "title": "Potato – Late blight",
        "description": "A fast-spreading disease (Phytophthora infestans) responsible for major historical famines.",
        "symptoms": [
            "Water-soaked, dark lesions on leaves and stems",
            "White fungal growth at lesion edges under humid conditions",
            "Brown, firm rot in tubers"
        ],
        "prevention": [
            "Plant certified disease-free seed tubers",
            "Destroy volunteer potatoes and cull piles",
            "Use resistant varieties where available",
            "Spray fungicides proactively during cool, wet weather"
        ],
    },
    "Potato___healthy": {
        "title": "Potato – Healthy leaf",
        "description": "Potato foliage with no blight or spotting.",
        "symptoms": [
            "Even green leaves without target-like or water-soaked lesions"
        ],
        "prevention": [
            "Follow crop rotation",
            "Hill soil well and manage irrigation to avoid prolonged leaf wetness"
        ],
    },
    "Raspberry___healthy": {
        "title": "Raspberry – Healthy leaf",
        "description": "Healthy raspberry foliage.",
        "symptoms": [
            "Uniform green lamina, no major spots or curling"
        ],
        "prevention": [
            "Plant in well-drained soil with good air flow",
            "Prune old canes and remove debris"
        ],
    },
    "Soybean___healthy": {
        "title": "Soybean – Healthy leaf",
        "description": "Soybean leaf without major disease symptoms.",
        "symptoms": [
            "Uniform green color, no rust pustules or blotches"
        ],
        "prevention": [
            "Rotate crops to reduce disease build-up",
            "Use high-quality, treated seed",
            "Scout routinely for rust and leaf spots"
        ],
    },
    "Squash___Powdery_mildew": {
        "title": "Squash – Powdery mildew",
        "description": "A fungal disease that commonly affects cucurbit leaves.",
        "symptoms": [
            "White powdery patches on upper leaf surfaces",
            "Premature yellowing and drying of leaves"
        ],
        "prevention": [
            "Plant in sunny, well-ventilated areas",
            "Avoid dense planting and excess nitrogen",
            "Use resistant varieties and fungicides as needed"
        ],
    },
    "Strawberry___Leaf_scorch": {
        "title": "Strawberry – Leaf scorch",
        "description": "A fungal leaf disease that can weaken plants over time.",
        "symptoms": [
            "Small purple spots that expand with tan centers",
            "Leaves may turn red-brown and dry out"
        ],
        "prevention": [
            "Use resistant cultivars where available",
            "Avoid overhead irrigation late in the day",
            "Remove and destroy old infected leaves after harvest"
        ],
    },
    "Strawberry___healthy": {
        "title": "Strawberry – Healthy leaf",
        "description": "Strawberry foliage without scorch or leaf spot.",
        "symptoms": [
            "Bright green trifoliate leaves with no necrotic spots"
        ],
        "prevention": [
            "Good spacing and mulch to reduce splash",
            "Rotate beds every few years"
        ],
    },
    "Tomato___Bacterial_spot": {
        "title": "Tomato – Bacterial spot",
        "description": "A bacterial disease affecting tomatoes and peppers.",
        "symptoms": [
            "Small water-soaked spots on leaves that turn dark and may drop out",
            "Raised brown spots on fruit, sometimes with cracking"
        ],
        "prevention": [
            "Use certified disease-free seed and transplants",
            "Avoid overhead irrigation and handling wet plants",
            "Rotate with non-host crops",
            "Use copper-based bactericides when conditions favor disease"
        ],
    },
    "Tomato___Early_blight": {
        "title": "Tomato – Early blight",
        "description": "A common fungal disease caused by Alternaria solani.",
        "symptoms": [
            "Brown leaf spots with concentric rings (target pattern)",
            "Yellowing and death of older leaves",
            "Dark sunken lesions at stem bases or on fruit shoulders"
        ],
        "prevention": [
            "Rotate crops and avoid planting tomatoes after potatoes or other solanaceae",
            "Stake or cage plants for better airflow",
            "Mulch soil to reduce splash",
            "Apply protectant fungicides as needed"
        ],
    },
    "Tomato___Late_blight": {
        "title": "Tomato – Late blight",
        "description": "A serious disease that can rapidly defoliate plants and rot fruit.",
        "symptoms": [
            "Water-soaked gray-green lesions that turn brown",
            "White moldy growth on undersides of leaves in humid weather",
            "Brown, firm lesions on fruit"
        ],
        "prevention": [
            "Destroy volunteer plants and cull piles",
            "Use resistant varieties when available",
            "Apply fungicides at first sign of disease or when weather is favorable"
        ],
    },
    "Tomato___Leaf_Mold": {
        "title": "Tomato – Leaf mold",
        "description": "A fungal disease favored by high humidity in greenhouses or dense canopies.",
        "symptoms": [
            "Yellow patches on upper leaf surfaces",
            "Olive-green to brown velvety mold on undersides",
            "Leaves may curl and die"
        ],
        "prevention": [
            "Improve ventilation and reduce humidity",
            "Avoid overhead watering late in the day",
            "Remove infected leaves promptly",
            "Use resistant varieties and fungicides if needed"
        ],
    },
    "Tomato___Septoria_leaf_spot": {
        "title": "Tomato – Septoria leaf spot",
        "description": "A foliar disease that can severely defoliate plants.",
        "symptoms": [
            "Numerous small, circular spots with dark borders and light centers",
            "Black fruiting bodies visible in lesion centers",
            "Lower leaves affected first, then upward"
        ],
        "prevention": [
            "Rotate crops and remove infected debris",
            "Mulch soil to reduce splash",
            "Apply protectant fungicides starting early in the season"
        ],
    },
    "Tomato___Spider_mites_Two-spotted_spider_mite": {
        "title": "Tomato – Two-spotted spider mite",
        "description": "An insect (mite) pest that causes stippling and bronzing of leaves.",
        "symptoms": [
            "Tiny yellow or white speckles on leaves",
            "Fine webbing on undersides of leaves",
            "Bronzed, dry foliage under heavy infestation"
        ],
        "prevention": [
            "Avoid drought stress; mites prefer hot, dry conditions",
            "Use strong water sprays to knock mites off plants",
            "Encourage natural predators (lady beetles, predatory mites)",
            "Use miticides only when thresholds are exceeded"
        ],
    },
    "Tomato___Target_Spot": {
        "title": "Tomato – Target spot",
        "description": "A fungal leaf spot disease that can also affect stems and fruit.",
        "symptoms": [
            "Circular brown spots with concentric rings (target-like)",
            "Lesions may have yellow halos",
            "Spots on fruit that can become sunken"
        ],
        "prevention": [
            "Use crop rotation and disease-free transplants",
            "Maintain good airflow by pruning lower leaves",
            "Apply fungicides when disease pressure is high"
        ],
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "title": "Tomato – Yellow leaf curl virus (TYLCV)",
        "description": "A viral disease transmitted mainly by whiteflies, causing severe yield loss.",
        "symptoms": [
            "Yellowing and upward curling of leaves",
            "Stunted plants with reduced internode length",
            "Poor fruit set and small fruit"
        ],
        "prevention": [
            "Control whiteflies with integrated pest management",
            "Use resistant/tolerant tomato varieties",
            "Remove infected plants to reduce virus sources",
            "Use insect-proof netting in seedling nurseries"
        ],
    },
    "Tomato___Tomato_mosaic_virus": {
        "title": "Tomato – Mosaic virus (ToMV/TMV)",
        "description": "A mechanically transmitted virus causing mottling and deformation.",
        "symptoms": [
            "Mottled light and dark green 'mosaic' on leaves",
            "Distorted, fern-like foliage",
            "Reduced plant vigor and yield"
        ],
        "prevention": [
            "Avoid smoking and handling plants without washing hands",
            "Disinfect tools and stakes regularly",
            "Use resistant varieties and certified seed",
            "Remove infected plants promptly"
        ],
    },
    "Tomato___healthy": {
        "title": "Tomato – Healthy leaf",
        "description": "Tomato foliage with no visible disease or pest injury.",
        "symptoms": [
            "Uniform green leaves with no spots, mold or curling"
        ],
        "prevention": [
            "Rotate crops and maintain good spacing",
            "Water at soil level rather than overhead",
            "Scout frequently for early signs of diseases and insects"
        ],
    },
    "Background_without_leaves": {
        "title": "Background without leaves",
        "description": "Images that do not contain plant leaves; used by the model as a background class.",
        "symptoms": [
            "No crop leaf present in the image"
        ],
        "prevention": [
            "If this is predicted for a leaf image, retake the photo with clearer focus on the leaf"
        ],
    },
}


# -----------------------------
# MODEL LOADING
# -----------------------------
MODEL_PATH = Path("best_resnet18.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(MODEL_PATH, map_location="cpu")

state = ckpt.get("model_state_dict", ckpt)
class2idx = ckpt["class2idx"]
idx2class = {v: k for k, v in class2idx.items()}
num_classes = len(idx2class)

model = models.resnet18(weights=None)
in_feats = model.fc.in_features
model.fc = nn.Linear(in_feats, num_classes)

if any(k.startswith("module.") for k in state.keys()):
    new_state = {k.replace("module.", ""): v for k, v in state.items()}
    state = new_state

model.load_state_dict(state, strict=False)
model.to(device).eval()

IMG_SIZE = int(ckpt.get("img_size", 224))
preprocess = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def overlay_label(img_pil, text):
    img = img_pil.copy().convert("RGBA")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", int(img.height * 0.05))
    except:
        font = ImageFont.load_default()

    # ---- FIX: SAFE TEXT SIZE CALCULATION ----
    try:
        # Works in new Pillow
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except Exception:
        try:
            # Works in older Pillow
            text_w, text_h = font.getsize(text)
        except:
            # Absolute fallback
            text_w, text_h = len(text) * 6, 12

    padding = 10

    # Draw background rectangle
    draw.rectangle(
        [0, 0, text_w + 2 * padding, text_h + 2 * padding],
        fill=(0, 0, 0, 150)
    )

    # Draw text
    draw.text(
        (padding, padding),
        text,
        font=font,
        fill=(255, 255, 255, 255)
    )

    return img.convert("RGB")


def predict_image(img):
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1).cpu().numpy().squeeze()

    top_idx = int(np.argmax(probs))
    return idx2class[top_idx]  # No confidence returned

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Crop Disease Detection", layout="wide")

# --- NAVBAR ---
selected = option_menu(
    menu_title=None,
    options=["Detection", "About Dataset"],
    icons=["camera", "info-circle"],
    orientation="horizontal",
)

# DIASES DATASET PATH
DATASET_PATH = Path("dataset")  # change to your dataset root folder path

# -----------------------------
# PAGE 1 — DETECTION
# -----------------------------
if selected == "Detection":

    st.title("🌿 Plant Disease Detection")
    uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", width=350)

        if st.button("Detect Disease"):
            label = predict_image(img)
            annotated_img = overlay_label(img, label)

            st.subheader("Result")
            st.image(annotated_img, caption=f"Detected: {label}", width=400)
            st.success(f"Disease: {label}")

            annotated_img.save("result.jpg")
            with open("result.jpg", "rb") as f:
                st.download_button("⬇ Download Result Image", f, "detected.jpg")

# -----------------------------
# PAGE 2 — ABOUT DATASET
# -----------------------------

elif selected == "About Dataset":
    st.title("📚 About the Dataset")
    st.write(
        "These are the classes detected by the model. "
        "Select any class to see disease details and prevention steps."
    )

    DATASET_PATH = Path("dataset")  # or your actual path as before

    if DATASET_PATH.exists():
        class_folders = sorted([f.name for f in DATASET_PATH.iterdir() if f.is_dir()])

        st.markdown(f"### Total Classes: **{len(class_folders)}**")

        # Build a pretty label for each folder
        pretty_labels = []
        for cls in class_folders:
            if cls == "Background_without_leaves":
                pretty = "Background without leaves"
            else:
                parts = cls.split("___")
                crop = parts[0].replace("_", " ")
                condition = parts[1].replace("_", " ") if len(parts) > 1 else ""
                pretty = f"{crop} → {condition}".strip()
            pretty_labels.append(pretty)

        # Let user choose one class
        selected_pretty = st.selectbox(
            "Select a disease / class:",
            options=pretty_labels,
            index=0,
        )

        # Map pretty name back to folder name
        idx = pretty_labels.index(selected_pretty)
        selected_class = class_folders[idx]

        info = disease_info.get(selected_class)

        if info:
            st.markdown(f"## {info['title']}")
            st.markdown(f"**Description:** {info['description']}")

            st.markdown("**Typical symptoms:**")
            for s in info["symptoms"]:
                st.markdown(f"- {s}")

            st.markdown("**Prevention / management steps:**")
            for p in info["prevention"]:
                st.markdown(f"- {p}")
        else:
            st.warning("Details for this class have not been added yet.")
    else:
        st.error("Dataset folder not found. Set DATASET_PATH correctly.")
