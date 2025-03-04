# Mode teacher forcing
def prepare_prompt(data_point, summary_included=True):
    prompt = f"Résume précisément le texte suivant en français en 100 mots maximum. Concentre-toi sur les points essentiels sans ajouter d'opinions ni de commentaires. Évite les phrases inutiles et reformule les idées clairement.\n\nTexte :\n{data_point['text']}\n\nRésumé concis et structuré (100 mots maximum) :"
    if summary_included:
        prompt+=f"\n\n{data_point['summary']}"
    return prompt