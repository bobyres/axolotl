from typing import Dict, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TextStreamer

model_name_or_path = "bobyres/Vigogne7b_finetune"
# model_name_or_path = "bofenghuang/vigogne-2-7b-chat"
# revision = "v2.0"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="right", use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto")

streamer = TextStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)


def chat(
    query: str,
    history: Optional[List[Dict]] = None,
    temperature: float = 0.7,
    top_p: float = 1.0,
    top_k: float = 0,
    repetition_penalty: float = 1.1,
    max_new_tokens: int = 1024,
    **kwargs,
):
    if history is None:
        history = []

    history.append({"role": "user", "content": query})

    input_ids = tokenizer.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt").to(model.device)
    input_length = input_ids.shape[1]

    generated_outputs = model.generate(
        input_ids=input_ids,
        generation_config=GenerationConfig(
            temperature=temperature,
            do_sample=temperature > 0.0,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            **kwargs,
        ),
        streamer=streamer,
        return_dict_in_generate=True,
    )

    generated_tokens = generated_outputs.sequences[0, input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    history.append({"role": "assistant", "content": generated_text})

    return generated_text, history

# Test sur un ensemble d'entretiens :
word_folder = os.path.abspath('C:/Generation_entretiens_3')
output_folder = os.path.abspath('C:/Compterendu_test')

# os.makedirs(output_folder, exist_ok=True)

# Parcourez tous les fichiers Word dans le dossier
for root, dirs, files in os.walk(word_folder):
    # Triez les fichiers par ordre numérique en prenant en compte le double soulignement
    files = sorted([f for f in files if f.endswith('.docx') and not f.startswith("~$")], key=lambda x: int(re.search(r'\d+', x).group()))
    
    for i, file in enumerate(files):
        if i < 2:
        # Chemin complet du fichier Word
            word_file_path = os.path.join(root, file)

            # Ouvrez le fichier Word
            doc = Document(word_file_path)

            # Extrait le texte du document Word
            text = "\n".join([para.text for para in doc.paragraphs])
            print(text)
            print("-------------------------------------------------------")
                    # # Générer un nom de fichier pour le compte rendu individuel
            output_file_name = f"Compte_rendu__{i}.docx"
            output_file_path = os.path.join(output_folder, output_file_name)
            

            prompt = f"""Tu dois réaliser un compte rendu le plus détaillé possible de : {text}"""
            # 1st round
            response, history = chat(prompt, history=None)

            print ("Compte rendu : ", '\n\n')
            print(response)
            # Créez un nouveau document Word
            document = Document()
            
            # Ajoutez la réponse au nouveau document
            document.add_paragraph(response)
            
            # Enregistrez le document Word avec le nom de fichier souhaité
            document.save(output_file_path)
