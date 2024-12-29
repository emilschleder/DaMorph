prompts = [
    {
        'category': 'Betingede Sætninger',
        'text': "Hvis jeg havde vidst, at det ville regne hele dagen, ville jeg have ",
        'generation_params': {
            'top_k': 30,           
            'top_p': 0.85,         
            'temperature': 0.7,    
            'category': 'Betingede Sætninger'
        }
    },
    {
        'category': 'Idiomatiske Udtryk',
        'text': "Efter at have arbejdet hårdt hele ugen, besluttede Peter sig for at tage tyren ved hornene og ",
        'generation_params': {
            'top_k': 40,
            'top_p': 0.9,
            'temperature': 0.8,
            'category': 'Idiomatiske Udtryk'
        }
    },
    {
        'category': 'Dialog med Emotioner',
        'text': "Maria: 'Hvorfor i alverden gjorde du det? Jeg kan ikke forstå, at du ' ",
        'generation_params': {
            'top_k': 30,
            'top_p': 0.85,
            'temperature': 0.8,
            'category': 'Dialog med Emotioner'
        }
    },
    {
        'category': 'Beskrivende Sprog',
        'text': "Den gamle kro ved havnen var kendt for sin hyggelige atmosfære og ",
        'generation_params': {
            'top_k': 40,
            'top_p': 0.9,
            'temperature': 0.85,
            'category': 'Beskrivende Sprog'
        }
    },
    {
        'category': 'Historisk Fortælling',
        'text': "For mange år siden, da jeg var en ung dreng, plejede jeg at ",
        'generation_params': {
            'top_k': 50,
            'top_p': 0.95,
            'temperature': 0.9,
            'category': 'Historisk Fortælling'
        }
    }
]