SemanticGustAVO
Riassunto modifiche

MODIFICHE ai CAMPI:

Prepara risposte e embedding

Classificazione intent con embedding

Chat endpoint

Avvio app

AGGIUNTA

Logica chatbot


Ho apportato delle migliorie al riguardo della strutturazione della logica embedding, con la tecnologia FAISS (Facebook AI Similarity Search) una libreria open source per la ricerca di similarità e il clustering di vettori
è stato aggiunta la possibilità di caricare un file intent.faiss con i dati emebdding elaborati senza doverli ricalcolare ad ogni avvio, se il file non esiste o non trovato il programma ne creerà uno nuovo e allo stesso tempo
verrà creato un intents_meta.json dove verranno caricati i metadati (cioè i tag e patterns) tutto estratto dal file intents.json.
Inoltre è stato estratto la logica di risposta dal @app(/chat) a una funzione indipendente per l'implementazione della modalità console dove la conversazione si terrà direttamente sulla console e quindi non arriva al frontend,
questa modalità l'ho aggiunta per facilitare i test locali senza dover caricare il index.html e altre cose.

Per avviare la modalità console digitare:
    python app.py --console
o semplicemente commentando tutto quello che ci sta dalla riga 229 alla 242 e decommentare i pezzi di codice dalla riga 244 alla 254

Per avviare la modalità standard digitare:
    python app.py

Dettagli modifiche

Riga 9 (Nuova aggiunta): Aggiunta l'import necessario per la libreria FAISS

Riga 25-27 (Nuova aggiunta): Estratto il caricamento delle risposte dal intent.json;
            Equivalente codice originale riga 36-38

Riga 36-90: Ristrutturazione della logica embedding per avere un salvataggio statico su disco dei dati elaborati emebedding, se non esistono ne verrà creata una all'avvio, 
            creazione intents_meta.json tag e patterns tutto estratto dal file intents.json al posto di averli insieme agli embedding come fatto al codice originale riga 40-47;
            Equivalente codice originale riga 34-47

Riga 102-118: Ristrutturazione di classificazione intent con embedding, la logica rimane la stessa si continua a utilizzare il cosine similarity;
              Equivalente codice originale riga 59-85

Riga 129-156 (Nuova aggiunta): Logica di risposta estratto dall'originale @app(/chat) per poter permettere l'implementazione della modalità console;
                               Equivalente codice originale riga 135-174

Riga 207-222: Ristrutturazione del app.route(/chat);
              
Riga 231-257: Implementazione della modalità console;
              Equivalente codice originale riga 181-182
