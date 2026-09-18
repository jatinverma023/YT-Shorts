"""
Transliteration module for YouTube Shorts captions.
Converts Devanagari Hindi words into clean, modern Roman Hindi (Hinglish)
while preserving English words, technical terms, punctuation, and 1:1 word timestamps.
"""
import re
import unicodedata
from typing import List, Dict, Any

from config import CAPTION_UPPERCASE, CAPTION_LANGUAGE_MODE


# Fast Unicode check for Devanagari block (U+0900 - U+097F)
DEVANAGARI_REGEX = re.compile(r"[\u0900-\u097F]")


def has_devanagari(text: str) -> bool:
    """Returns True if the string contains any Devanagari characters."""
    if not text:
        return False
    return bool(DEVANAGARI_REGEX.search(text))


# Comprehensive curated dictionary for conversational Hindi / Hinglish words
DEVANAGARI_COMMON_WORDS = {
    # Pronouns & determiners
    "मैं": "main",
    "मै": "main",
    "मुझे": "mujhe",
    "मुझको": "mujhko",
    "मेरा": "mera",
    "मेरी": "meri",
    "मेरे": "mere",
    "हम": "hum",
    "हमें": "humein",
    "हमको": "humko",
    "हमारा": "hamara",
    "हमारी": "hamari",
    "हमारे": "hamare",
    "तू": "tu",
    "तुझे": "tujhe",
    "तेरा": "tera",
    "तेरी": "teri",
    "तेरे": "tere",
    "तुम": "tum",
    "तुम्हें": "tumhein",
    "तुमको": "tumko",
    "तुम्हारा": "tumhara",
    "तुम्हारी": "tumhari",
    "तुम्हारे": "tumhare",
    "आप": "aap",
    "आपको": "aapko",
    "आपका": "aapka",
    "आपकी": "aapki",
    "आपके": "aapke",
    "वह": "woh",
    "वो": "woh",
    "उस": "us",
    "उसने": "usne",
    "उसे": "use",
    "उसको": "usko",
    "उसका": "uska",
    "उसकी": "uski",
    "उसके": "uske",
    "यह": "yeh",
    "ये": "ye",
    "इस": "is",
    "इसने": "isne",
    "इसे": "ise",
    "इसको": "isko",
    "इसका": "iska",
    "इसकी": "iski",
    "इसके": "iske",
    "वे": "ve",
    "उन": "un",
    "उन्होंने": "unhone",
    "उन्हें": "unhein",
    "उनको": "unko",
    "उनका": "unka",
    "उनकी": "unki",
    "उनके": "unke",
    "इन": "in",
    "इन्होंने": "inhone",
    "इन्हें": "inhein",
    "इनको": "inko",
    "इनका": "inka",
    "इनकी": "inki",
    "इनके": "inke",
    "अपना": "apna",
    "अपनी": "apni",
    "अपने": "apne",
    "खुद": "khud",
    "स्वयं": "swayam",
    "कोई": "koi",
    "कुछ": "kuch",
    "किसी": "kisi",
    "सब": "sab",
    "सभी": "sabhi",
    "सारे": "saare",
    "सारी": "saari",
    "सारा": "saara",

    # Conjunctions, particles, prepositions
    "और": "aur",
    "तथा": "tatha",
    "एवं": "evam",
    "या": "ya",
    "अथवा": "athwa",
    "लेकिन": "lekin",
    "मगर": "magar",
    "किन्तु": "kintu",
    "परंतु": "parantu",
    "पर": "par",
    "पे": "pe",
    "में": "mein",
    "से": "se",
    "को": "ko",
    "का": "ka",
    "की": "ki",
    "के": "ke",
    "लिए": "liye",
    "तक": "tak",
    "साथ": "saath",
    "बिना": "bina",
    "अगर": "agar",
    "यदि": "yadi",
    "तो": "toh",
    "ताकि": "taaki",
    "क्योंकि": "kyunki",
    "इसलिए": "isliye",
    "भी": "bhi",
    "ही": "hi",

    # Question words
    "क्या": "kya",
    "क्यों": "kyun",
    "क्यो": "kyun",
    "कब": "kab",
    "कहाँ": "kahan",
    "कहा": "kahan",
    "कैसे": "kaise",
    "कैसा": "kaisa",
    "कैसी": "kaisi",
    "कौन": "kaun",
    "किसे": "kise",
    "किसको": "kisko",
    "कितना": "kitna",
    "कितने": "kitne",
    "कितनी": "kitni",

    # Auxiliary verbs & forms
    "है": "hai",
    "हैं": "hain",
    "हूँ": "hoon",
    "हु": "hoon",
    "हो": "ho",
    "था": "tha",
    "थी": "thi",
    "थे": "the",
    "होगी": "hogi",
    "होगा": "hoga",
    "होंगे": "honge",
    "हुआ": "hua",
    "हुई": "hui",
    "हुए": "hue",
    "होता": "hota",
    "होती": "hoti",
    "होते": "hote",
    "होना": "hona",
    "होने": "hone",
    "रहा": "raha",
    "रही": "rahi",
    "रहे": "rahe",
    "रहना": "rahna",
    "रहते": "rahte",
    "रहता": "rahta",
    "रहती": "rahti",
    "कर": "kar",
    "करना": "karna",
    "करने": "karne",
    "करनी": "karni",
    "करता": "karta",
    "करते": "karte",
    "करती": "karti",
    "किया": "kiya",
    "किये": "kiye",
    "किए": "kiye",
    "की": "ki",
    "करो": "karo",
    "करें": "karein",
    "करेंगे": "karenge",
    "करेगा": "karega",
    "करेगी": "karegi",

    # Common verbs
    "सीख": "seekh",
    "सीखना": "seekhna",
    "सीखने": "seekhne",
    "सीखनी": "seekhni",
    "सीखता": "seekhta",
    "सीखते": "seekhte",
    "सीखती": "seekhti",
    "सीखा": "seekha",
    "सीखे": "seekhe",
    "सीखो": "seekho",
    "भाग": "bhaag",
    "भागना": "bhaagna",
    "भागने": "bhaagne",
    "भागते": "bhaagte",
    "भागता": "bhaagta",
    "भागती": "bhaagti",
    "भागा": "bhaaga",
    "भागे": "bhaage",
    "देख": "dekh",
    "देखना": "dekhna",
    "देखने": "dekhne",
    "देखता": "dekhta",
    "देखते": "dekhte",
    "देखती": "dekhti",
    "देखा": "dekha",
    "देखे": "dekhe",
    "देखो": "dekho",
    "देखें": "dekhein",
    "बोल": "bol",
    "बोलना": "bolna",
    "बोलने": "bolne",
    "बोलता": "bolta",
    "बोलते": "bolte",
    "बोलती": "bolti",
    "बोला": "bola",
    "बोले": "bole",
    "बोलो": "bolo",
    "सुन": "sun",
    "सुनना": "sunna",
    "सुनने": "sunne",
    "सुनता": "sunta",
    "सुनते": "sunte",
    "सुनती": "sunti",
    "सुना": "suna",
    "सुने": "sune",
    "सुनो": "suno",
    "जा": "jaa",
    "जाना": "jaana",
    "जाने": "jaane",
    "जाता": "jaata",
    "जाते": "jaate",
    "जाती": "jaati",
    "गया": "gaya",
    "गए": "gaye",
    "गई": "gayi",
    "जाओ": "jaao",
    "जाएं": "jaayein",
    "आ": "aa",
    "आना": "aana",
    "आने": "aane",
    "आता": "aata",
    "आते": "aate",
    "आती": "aati",
    "आया": "aaya",
    "आए": "aaye",
    "आई": "aayi",
    "आओ": "aao",
    "आएं": "aayein",
    "ले": "le",
    "लेना": "lena",
    "लेने": "lene",
    "लेता": "leta",
    "लेते": "lete",
    "लेती": "leti",
    "लिया": "liya",
    "लिए": "liye",
    "लो": "lo",
    "लें": "lein",
    "दे": "de",
    "देना": "dena",
    "देने": "dene",
    "देता": "deta",
    "देते": "dete",
    "देती": "deti",
    "दिया": "diya",
    "दिए": "diye",
    "दी": "di",
    "दो": "do",
    "दें": "dein",
    "समझ": "samajh",
    "समझना": "samajhna",
    "समझने": "samajhne",
    "समझता": "samajhta",
    "समझते": "samajhte",
    "समझती": "samajhti",
    "समझा": "samjha",
    "समझे": "samjhe",
    "समझो": "samjho",
    "सोच": "soch",
    "सोचना": "sochna",
    "सोचता": "sochta",
    "सोचते": "sochte",
    "सोचा": "socha",
    "सोचो": "socho",
    "चल": "chal",
    "चलना": "chalna",
    "चलते": "chalte",
    "चलता": "chalta",
    "चलो": "chalo",
    "चला": "chala",
    "चले": "chale",
    "मिल": "mil",
    "मिलना": "milna",
    "मिलता": "milta",
    "मिलते": "milte",
    "मिला": "mila",
    "मिले": "mile",
    "रख": "rakh",
    "रखना": "rakhna",
    "रखता": "rakhta",
    "रखा": "rakha",
    "रखो": "rakho",

    # Common adverbs & adjectives
    "मत": "mat",
    "नहीं": "nahi",
    "नही": "nahi",
    "ना": "na",
    "हाँ": "haan",
    "हा": "haan",
    "बहुत": "bahut",
    "ज्यादा": "zyada",
    "कम": "kam",
    "थोड़ा": "thoda",
    "थोड़े": "thode",
    "थोड़ी": "thodi",
    "पूरा": "poora",
    "पूरे": "poore",
    "पूरी": "poori",
    "सिर्फ": "sirf",
    "बस": "bas",
    "केवल": "kewal",
    "हमेशा": "hamesha",
    "कभी": "kabhi",
    "अभी": "abhi",
    "तभी": "tabhi",
    "जब": "jab",
    "तब": "tab",
    "फिर": "phir",
    "दोबारा": "dobaara",
    "बार": "baar",
    "पहले": "pehle",
    "बाद": "baad",
    "आगे": "aage",
    "पीछे": "peeche",
    "ऊपर": "upar",
    "नीचे": "neeche",
    "अंदर": "andar",
    "बाहर": "baahar",
    "यहाँ": "yahan",
    "वहाँ": "wahan",
    "जहाँ": "jahan",
    "सच": "sach",
    "झूठ": "jhooth",
    "सही": "sahi",
    "गलत": "galat",
    "अच्छा": "achha",
    "अच्छे": "achhe",
    "अच्छी": "achhi",
    "बुरा": "bura",
    "बुरे": "bure",
    "बुरी": "buri",
    "बड़ा": "bada",
    "बड़े": "bade",
    "बड़ी": "badi",
    "छोटा": "chhota",
    "छोटे": "chhote",
    "छोटी": "chhoti",
    "नया": "naya",
    "नए": "naye",
    "नई": "nayi",
    "पुराना": "purana",
    "पुराने": "purane",

    # Common nouns & Hindi podcast words
    "लोग": "log",
    "लोगों": "logon",
    "बात": "baat",
    "बातें": "baatein",
    "बातों": "baaton",
    "काम": "kaam",
    "समय": "samay",
    "वक़्त": "waqt",
    "वक्त": "waqt",
    "ज़िंदगी": "zindagi",
    "जिंदगी": "zindagi",
    "जीवन": "jeevan",
    "दोस्त": "dost",
    "दोस्तों": "doston",
    "यार": "yaar",
    "भाई": "bhai",
    "दिन": "din",
    "रात": "raat",
    "साल": "saal",
    "महीना": "maheena",
    "महीने": "maheene",
    "पैसा": "paisa",
    "पैसे": "paise",
    "पैसों": "paison",
    "दुनिया": "duniya",
    "देश": "desh",
    "भारत": "bharat",
    "नाम": "naam",
    "सवाल": "sawaal",
    "जवाब": "jawab",
    "तरीका": "tareeka",
    "तरीके": "tareeke",
    "चीज": "cheez",
    "चीजें": "cheezein",
    "गलती": "galti",
    "गलतियां": "galtiyan",

    # English words frequently written in Devanagari by Whisper
    "कोडिंग": "coding",
    "प्रोग्रामिंग": "programming",
    "कंप्यूटर": "computer",
    "इंटरनेट": "internet",
    "सॉफ्टवेयर": "software",
    "हार्डवेयर": "hardware",
    "स्टूडेंट": "student",
    "स्टूडेंट्स": "students",
    "छात्र": "chhatra",
    "डिग्री": "degree",
    "कॉलेज": "college",
    "स्कूल": "school",
    "यूनिवर्सिटी": "university",
    "जॉब": "job",
    "ऑफिस": "office",
    "कंपनी": "company",
    "बिजनेस": "business",
    "स्टार्टअप": "startup",
    "चैनल": "channel",
    "वीडियो": "video",
    "यूट्यूब": "youtube",
    "सब्सक्राइब": "subscribe",
    "लाइक": "like",
    "शेयर": "share",
    "कमेंट": "comment",
    "फॉलो": "follow",
    "मिस्टेक": "mistake",
    "प्रॉब्लम": "problem",
    "सॉल्यूशन": "solution",
    "रिजल्ट": "result",
    "सक्सेस": "success",
    "फेलियर": "failure",
    "टाइम": "time",
    "लाइफ": "life",
    "इंडिया": "india",
    "सर": "sir",
    "मैम": "maam",
    "कंसिस्टेंसी": "consistency",
    "फोकस": "focus",
    "पॉडकास्ट": "podcast",
    "शॉर्ट्स": "shorts",
    "शॉर्ट": "short",
    "क्लिप": "clip",
    "आर्टिफिशियल": "artificial",
    "इंटेलिजेंस": "intelligence",
    "एआई": "AI",
    "कंटेंट": "content",
    "क्रिएटर": "creator",
    "ऑनलाइन": "online",
    "ऑफलाइन": "offline",
    "मोबाइल": "mobile",
    "फोन": "phone",
}


# Phonetic mappings for fallback transliteration
VOWELS = {
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo',
    'ऋ': 'ri', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
    'ऑ': 'o', 'ऍ': 'e',
}

MATRAS = {
    'ा': 'aa', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo',
    'ृ': 'ri', 'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
    'ॉ': 'o', 'ॅ': 'e',
}

CONSONANTS = {
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v',
    'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
    # Nukta consonants
    'क़': 'q', 'ख़': 'kh', 'ग़': 'gh', 'ज़': 'z', 'ड़': 'd', 'ढ़': 'dh', 'फ़': 'f',
    'य़': 'y', 'ऩ': 'n', 'ऱ': 'r',
}

VIRAMA = '्'
ANUSVARA = 'ं'
CHANDRABINDU = 'ँ'
VISARGA = 'ः'
NUKTA = '़'


def transliterate_devanagari_word_phonetic(word: str) -> str:
    """
    Phonetically transliterates a Devanagari Hindi word into clean Roman text
    using standard Hindi pronunciation rules including schwa deletion.
    """
    chars = list(word)
    n = len(chars)
    result = []
    i = 0

    while i < n:
        c = chars[i]
        next_c = chars[i + 1] if i + 1 < n else None

        # Check for Nukta combining (e.g. क + ़ = क़)
        if next_c == NUKTA:
            combined = c + next_c
            if combined in CONSONANTS:
                c = combined
                i += 1
                next_c = chars[i + 1] if i + 1 < n else None

        if c in CONSONANTS:
            base_cons = CONSONANTS[c]
            # Check what follows
            if next_c == VIRAMA:
                # Halant: pure consonant, no vowel
                result.append(base_cons)
                i += 2  # skip consonant and virama
                continue
            elif next_c in MATRAS:
                # Matra follows
                result.append(base_cons + MATRAS[next_c])
                i += 2  # skip consonant and matra
                continue
            elif next_c in (ANUSVARA, CHANDRABINDU):
                # Anusvara or Chandrabindu right after consonant
                result.append(base_cons + "an")
                i += 2
                continue
            else:
                # Consonant followed by another consonant or end of word (schwa deletion rule)
                if i + 1 >= n:
                    # Final consonant: schwa is dropped in Hindi (e.g., कर -> kar)
                    result.append(base_cons)
                else:
                    # Medial consonant
                    result.append(base_cons + "a")
                i += 1
                continue

        elif c in VOWELS:
            v_val = VOWELS[c]
            if next_c in (ANUSVARA, CHANDRABINDU):
                v_val += "n"
                i += 1
            result.append(v_val)
            i += 1
            continue

        elif c in MATRAS:
            # Standalone matra (rare error in raw text)
            result.append(MATRAS[c])
            i += 1
            continue

        elif c == ANUSVARA:
            # Preceding vowel nasalization
            result.append("n")
            i += 1
            continue

        elif c == CHANDRABINDU:
            result.append("n")
            i += 1
            continue

        elif c == VISARGA:
            result.append("h")
            i += 1
            continue

        elif c == VIRAMA:
            # Already handled with consonant
            i += 1
            continue

        else:
            # Any non-Devanagari character (digits, Latin, punctuation, symbols)
            result.append(c)
            i += 1
            continue

    raw_output = "".join(result)

    # Clean up awkward double vowels or unnatural sequences from rule-based conversion
    cleaned = (
        raw_output
        .replace("aan", "an")
        .replace("aaan", "an")
        .replace("ein", "ein")
        .replace("iin", "een")
    )
    return cleaned


def romanize_single_word(word_str: str, uppercase: bool = True) -> str:
    """
    Romanizes an individual word string.
    Preserves leading and trailing punctuation (e.g., '"नमस्ते!"' -> '"NAMASTE!"').
    Preserves pure English / Latin words without alteration.
    Transliterates Devanagari Hindi words via dictionary or phonetic engine.
    """
    if not word_str:
        return ""

    # Check if contains Devanagari
    if not has_devanagari(word_str):
        # Already English / Latin or numeric
        return word_str.upper() if uppercase else word_str

    # Extract leading/trailing non-Devanagari punctuation
    match = re.match(r"^([^a-zA-Z0-9\u0900-\u097F]*)([\u0900-\u097F\w\-]+)([^a-zA-Z0-9\u0900-\u097F]*)$", word_str)
    if not match:
        core = word_str
        prefix = ""
        suffix = ""
    else:
        prefix, core, suffix = match.groups()

    # Normalize core string for lookup
    core_clean = unicodedata.normalize("NFC", core.strip())

    # Check curated dictionary first
    if core_clean in DEVANAGARI_COMMON_WORDS:
        trans = DEVANAGARI_COMMON_WORDS[core_clean]
    elif core_clean.lower() in DEVANAGARI_COMMON_WORDS:
        trans = DEVANAGARI_COMMON_WORDS[core_clean.lower()]
    else:
        # Fallback to phonetic transliteration
        trans = transliterate_devanagari_word_phonetic(core_clean)

    out_word = f"{prefix}{trans}{suffix}"
    return out_word.upper() if uppercase else out_word


def romanize_words(words: List[Dict[str, Any]], uppercase: bool = None) -> List[Dict[str, Any]]:
    """
    Transforms a list of Whisper word dictionaries into Roman Hindi / Hinglish.
    Guarantees strict 1:1 mapping:
    - len(output) == len(words)
    - exact same 'start' and 'end' timestamps on every single word
    - pure English words are kept English
    - Devanagari words are converted to clean Roman Hindi
    - words are uppercased if uppercase=True (default from config)
    """
    if not words:
        return []

    if uppercase is None:
        uppercase = CAPTION_UPPERCASE

    # If caption mode is disabled or "original", return words with just uppercase applied
    if CAPTION_LANGUAGE_MODE == "original":
        return [
            {**w, "word": (w.get("word", "").upper() if uppercase else w.get("word", ""))}
            for w in words
        ]

    romanized = []
    for w in words:
        orig_word = str(w.get("word", "")).strip()
        new_word = romanize_single_word(orig_word, uppercase=uppercase)
        # Deep copy all other fields including start, end, and any custom metadata
        w_new = dict(w)
        w_new["word"] = new_word
        romanized.append(w_new)

    return romanized
