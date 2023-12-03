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


entretien = """ Question 1: Pouvez-vous vous présenter en quelques mots?
Réponse 1: Bonjour, je suis Alexandre et je suis ravi de discuter avec vous aujourd'hui au sujet du poste de Statistical Project Lead Biomarker chez Sanofi. J'ai une solide formation en statistiques, avec un doctorat en biostatistique de l'Université Pierre et Marie Curie à Paris. Ma passion pour les statistiques appliquées à la recherche médicale m'a conduit à poursuivre une carrière dans ce domaine.
Question 2: Parlez-moi de votre formation? Pourquoi avoir choisi cette formation et quelles compétences vous a-t-elle permis d'avoir?
Réponse 2: J'ai choisi de me spécialiser en biostatistique parce que j'ai toujours été fasciné par la manière dont les données peuvent être utilisées pour améliorer la santé et la qualité de vie des gens. Ma formation m'a permis de développer des compétences avancées en modélisation statistique, en analyse de données longitudinales, et en conception d'essais cliniques. J'ai également acquis une expertise dans l'utilisation de logiciels tels que SAS, R, et STATA pour l'analyse de données complexes en biomédecine.
Question 3: Quel est votre niveau d'expérience? Quel(s) poste(s) avez-vous occupé et dans quelle entreprise?
Réponse 3: J'ai accumulé plus de dix ans d'expérience en tant que biostatisticien dans plusieurs entreprises et institutions de renom. J'ai commencé ma carrière en tant que statisticien chez l'Institut National de la Santé et de la Recherche Médicale (INSERM), où j'ai travaillé sur des projets de recherche en épidémiologie. Par la suite, j'ai rejoint le groupe de recherche en biostatistique de l'Université de Paris-Sud en tant que chercheur principal, où j'ai dirigé des équipes de statisticiens sur des projets de grande envergure.
J'ai également eu l'opportunité de travailler en tant que biostatisticien principal chez Sanofi, où j'ai dirigé l'analyse statistique de plusieurs essais cliniques majeurs dans le domaine des maladies cardiovasculaires. Mon expertise dans l'analyse de données biomédicales et ma capacité à collaborer avec des chercheurs et des cliniciens font de moi un candidat idéal pour le poste de Statistical Project Lead Biomarker.
Question 4: Quels sont les outils/logiciels que vous maîtrisez?
Réponse 4: J'ai une maîtrise avancée de plusieurs logiciels statistiques, y compris SAS, R, STATA, et Python. J'ai également une grande expérience dans l'utilisation de bases de données complexes pour l'analyse de données longitudinales et spatiales. Ma capacité à programmer des analyses statistiques personnalisées me permet de résoudre des problèmes complexes et de fournir des résultats de haute qualité.
Question 5: Quelle est votre situation actuelle?
Réponse 5: Actuellement, je suis le responsable de l'unité de biostatistique avancée chez l'INSERM, où je supervise une équipe de statisticiens travaillant sur des projets de recherche en épidémiologie et en biomédecine. Mon rôle consiste à assurer la qualité des analyses statistiques, à collaborer avec des chercheurs et des cliniciens, et à contribuer à la conception d'essais cliniques. Cependant, je suis à la recherche de nouveaux défis et je suis très intéressé par l'opportunité de rejoindre Sanofi en tant que Statistical Project Lead Biomarker pour apporter mon expertise à des projets innovants dans le domaine de la recherche médicale.
Question 6: Pour finir notre entretien quelles seraient vos disponibilités? Avez-vous un préavis?
Réponse 6: Je suis prêt à commencer dès que possible et je n'ai actuellement aucun préavis à respecter. Je suis enthousiaste à l'idée de rejoindre Sanofi et de contribuer à des projets qui ont un impact significatif sur la santé des patients. Je suis flexible en termes de disponibilité pour faciliter la transition vers ce nouveau rôle passionnant. """

prompt = f"""Tu dois réaliser un compte rendu le plus détaillé possible de : {entretien}"""
# 1st round
response, history = chat(prompt, history=None)
