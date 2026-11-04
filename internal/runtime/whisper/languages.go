package whisper

import "strings"

// languageNames are Whisper's language codes and their English names (openai/whisper's LANGUAGES, and transformers' TO_LANGUAGE_CODE read the other way). A checkpoint's own lang_to_id decides which of them
// it can use; large-v3 adds Cantonese.
var languageNames = map[string]string{
	"en":  "english",
	"zh":  "chinese",
	"de":  "german",
	"es":  "spanish",
	"ru":  "russian",
	"ko":  "korean",
	"fr":  "french",
	"ja":  "japanese",
	"pt":  "portuguese",
	"tr":  "turkish",
	"pl":  "polish",
	"ca":  "catalan",
	"nl":  "dutch",
	"ar":  "arabic",
	"sv":  "swedish",
	"it":  "italian",
	"id":  "indonesian",
	"hi":  "hindi",
	"fi":  "finnish",
	"vi":  "vietnamese",
	"he":  "hebrew",
	"uk":  "ukrainian",
	"el":  "greek",
	"ms":  "malay",
	"cs":  "czech",
	"ro":  "romanian",
	"da":  "danish",
	"hu":  "hungarian",
	"ta":  "tamil",
	"no":  "norwegian",
	"th":  "thai",
	"ur":  "urdu",
	"hr":  "croatian",
	"bg":  "bulgarian",
	"lt":  "lithuanian",
	"la":  "latin",
	"mi":  "maori",
	"ml":  "malayalam",
	"cy":  "welsh",
	"sk":  "slovak",
	"te":  "telugu",
	"fa":  "persian",
	"lv":  "latvian",
	"bn":  "bengali",
	"sr":  "serbian",
	"az":  "azerbaijani",
	"sl":  "slovenian",
	"kn":  "kannada",
	"et":  "estonian",
	"mk":  "macedonian",
	"br":  "breton",
	"eu":  "basque",
	"is":  "icelandic",
	"hy":  "armenian",
	"ne":  "nepali",
	"mn":  "mongolian",
	"bs":  "bosnian",
	"kk":  "kazakh",
	"sq":  "albanian",
	"sw":  "swahili",
	"gl":  "galician",
	"mr":  "marathi",
	"pa":  "punjabi",
	"si":  "sinhala",
	"km":  "khmer",
	"sn":  "shona",
	"yo":  "yoruba",
	"so":  "somali",
	"af":  "afrikaans",
	"oc":  "occitan",
	"ka":  "georgian",
	"be":  "belarusian",
	"tg":  "tajik",
	"sd":  "sindhi",
	"gu":  "gujarati",
	"am":  "amharic",
	"yi":  "yiddish",
	"lo":  "lao",
	"uz":  "uzbek",
	"fo":  "faroese",
	"ht":  "haitian creole",
	"ps":  "pashto",
	"tk":  "turkmen",
	"nn":  "nynorsk",
	"mt":  "maltese",
	"sa":  "sanskrit",
	"lb":  "luxembourgish",
	"my":  "myanmar",
	"bo":  "tibetan",
	"tl":  "tagalog",
	"mg":  "malagasy",
	"as":  "assamese",
	"tt":  "tatar",
	"haw": "hawaiian",
	"ln":  "lingala",
	"ha":  "hausa",
	"ba":  "bashkir",
	"jw":  "javanese",
	"su":  "sundanese",
	"yue": "cantonese",
}

// LanguageCode resolves what a client wrote for a language ("en", "English", "<|en|>") to its code, or "" when it is none Whisper knows.
func LanguageCode(s string) string {
	s = strings.ToLower(strings.TrimSpace(strings.Trim(strings.TrimSpace(s), "<|>")))
	if _, ok := languageNames[s]; ok {
		return s
	}
	for code, name := range languageNames {
		if name == s {
			return code
		}
	}
	return ""
}

// LanguageName is the English name of a language token ("<|en|>") or code ("en"): "english"; the code itself when unknown.
func LanguageName(tokenOrCode string) string {
	code := strings.Trim(tokenOrCode, "<|>")
	if n, ok := languageNames[code]; ok {
		return n
	}
	return code
}
