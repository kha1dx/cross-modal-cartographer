#!/usr/bin/env python3
"""
build_registry.py
-----------------
Auto-derives landmark metadata (type, era, region, city, style, coordinates)
from the dataset folder names and produces dataset/landmarks_registry.csv.

Run from V0/:
    python build_registry.py
"""

import os
import csv

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "dataset", "images")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "..", "dataset", "landmarks_registry.csv")

# ── Explicit overrides (checked BEFORE substring rules) ──────────────────────
# Use this for landmarks whose names cause substring collisions or whose
# classification cannot be inferred from the name alone.
TYPE_OVERRIDES = {
    # "bay" in "bayt" collision
    "Bayt_Al-Suhaymi":                      "Historic House",
    # "garden" in "garden_city" collision
    "Garden_City,_Cairo":                    "Neighborhood",
    # "island" in name but it's a measuring instrument
    "Nilometer_in_Rhoda_Island":             "Monument",
    # "park" but it's a nature reserve
    "Wadi_el_Gemal_National_Park":           "Natural Site",
    # "montaza" triggers Garden/Park but it's a palace
    "Montaza_Palace":                        "Palace",
    # "plateau" triggers Natural Site but it's an archaeological site
    "Giza_Plateau":                          "Archaeological Site",
    # "giza_pyramid_complex" triggers Pyramid but it's a complex
    "Giza_pyramid_complex":                  "Complex",
    # Gezira triggers Neighborhood but this is a museum
    "Gezira_Center_for_Modern_Art":          "Museum",
    # Specific landmarks whose type can't be guessed from name
    "Colossi_of_Memnon":                     "Monument",
    "Great_Sphinx_of_Giza":                  "Monument",
    "Sphinx_of_Memphis":                     "Monument",
    "Pompey's_Pillar,_Alexandria":           "Monument",
    "Port_Tewfik_Memorial":                  "Monument",
    "Relations_of_Egypt_and_Italy":          "Monument",
    "Tomb_of_Unknown_Soldier_in_Cairo":      "Monument",
    "Sabil_of_Abd_al-Rahman_Katkhuda":       "Fountain",
    "San_Stefano_Grand_Plaza":               "Hotel",
    "Shepheard's_Hotel":                     "Hotel",
    "Sha'b_Abu_el-Nuhas":                   "Natural Site",
    "Alexandria_Port":                       "Infrastructure",
    "Children's_Civilisation_and_Creativity_Centre": "Museum",
    "Kom_el-Shoqafa":                        "Tomb",
    # Temples misclassified as Historical Site
    "Amada":                                 "Temple",
    "Medinet_Madi":                          "Temple",
    "Qasr_Qarun":                            "Temple",
    "Kiosk_of_Qertassi":                     "Temple",
    "Kiosk_of_Trajan_in_Philae":             "Temple",
    "New_Kalabsha":                          "Temple",
    # Pyramid missing from rules
    "Meidum":                                "Pyramid",
    # Tombs missing from rules
    "Beni_Hassan":                           "Tomb",
    # GLD-specific overrides (different transliterations)
    "The_Great_Sphinx":                      "Monument",
    "Dayr_al_Madīnah":                       "Archaeological Site",
    "Kawm_Madinat_Madi":                     "Temple",
    "Gerf_Hussein":                          "Temple",
    "The_First_Terrace":                     "Temple",
    "Fustat":                                "Archaeological Site",
    "Whale_Valley":                          "Natural Site",
    "Unknown_Soldier_Memorial":              "Monument",
    "Qaitbay_Citadel_Street":                "Street",
    "Red_Sea":                               "Natural Site",
    "Pylon":                                 "Temple",
    "Court":                                 "Temple",
    "Pavillion":                             "Temple",
    "Valley_of_the_nobles_Asasif":           "Tomb",
    "Archaeological_Park___Pompey_s_Pillar": "Monument",
    "Funerary_Temple_of_Khafre":             "Temple",
    "Coptic_Cairo":                          "Neighborhood",
    "National_Cultural_Centre":              "Museum",
    "Egyptian_Theater":                      "Theater",
    "Egyptian__Spice__Bazaar":               "Market",
    "Winter_Palace":                         "Hotel",
    "Kalabsha_City":                         "Temple",
    "Al-Azhar_Park_(Cairo)":                 "Garden/Park",
    # GLD vague entries
    "Al_Minya":                              "Neighborhood",
    "Aswan":                                 "Neighborhood",
    "This_appears_to_be_related_to_Abu_Rawash": "Archaeological Site",
    "This_appears_to_be_related_to_Ramleh_Station": "Infrastructure",
}

ERA_OVERRIDES = {
    # Modern infrastructure / museums wrongly defaulting to Islamic
    "Aswan_High_Dam":                        "Contemporary",
    "Aswan_Low_Dam":                         "Contemporary",
    "Aswan_Museum":                          "Contemporary",
    "Aswan_Botanical_Garden":                "Contemporary",
    "Ahmed_Shawki_Museum":                   "Contemporary",
    "Alexandria_National_Museum":            "Contemporary",
    "Alexandria_Zoo":                        "Contemporary",
    "Aquarium_Grotto_Garden":                "Contemporary",
    "Egyptian_Museum_(Cairo)":               "Contemporary",
    "Egyptian_Geological_Museum":            "Contemporary",
    "Egyptian_National_Library":             "Contemporary",
    "Egyptian_National_Military_Museum":     "Contemporary",
    "El_Alamein_Military_Museum":            "Contemporary",
    "Giza_Zoo":                              "Contemporary",
    "Giza_Plateau":                          "Pharaonic",
    "Grand_Egyptian_Museum":                 "Contemporary",
    "Hurghada_Grand_Aquarium":               "Contemporary",
    "Luxor_Museum":                          "Contemporary",
    "Mallawi_Museum":                        "Contemporary",
    "Nubia_Museum,_Aswan":                   "Contemporary",
    "Orman_Garden":                          "Contemporary",
    "Sadat_museum":                          "Contemporary",
    "Umm_Kulthum_Museum":                    "Contemporary",
    "Corniche_(Alexandria)":                 "Contemporary",
    "EL-Safa_Park":                          "Contemporary",
    "Shalalat_Garden,_Alexandria":           "Contemporary",
    "Port_Said_Lighthouse":                  "Contemporary",
    "Qasr_al-Nil_Bridge":                    "Contemporary",
    "Tomb_of_Unknown_Soldier_in_Cairo":      "Contemporary",
    "Port_Tewfik_Memorial":                  "Contemporary",
    "Relations_of_Egypt_and_Italy":          "Contemporary",
    # Synagogues wrongly falling to Islamic
    "Eliyahu_Hanavi_Synagogue_(Alexandria)": "Contemporary",
    "Sha'ar_Hashamayim_Synagogue_(Cairo)":   "Contemporary",
    "Synagogue_of_Moses_Maimonides":         "Medieval",
    "Ben_Ezra_Synagogue":                    "Medieval",
    # Wrong keyword matches
    "Greco-Roman_Museum,_Alexandria":        "Greco-Roman",
    "Mosque_of_Saint_Ibrahim_El-Desouky":    "Islamic",
    "Bagawat":                               "Coptic/Byzantine",
    "Faiyum_Zoo":                            "Contemporary",
    "Antoniadis_Palace":                     "Contemporary",
    "Museum_of_Constantine_P._Cavafy":       "Contemporary",
    "Kiosk_of_Trajan_in_Philae":             "Greco-Roman",
    "Gilf_Kebir_Plateau":                    "Prehistoric",
    "Montaza_Palace":                        "Contemporary",
    # Natural sites — no human era
    "Abu_Qir_Bay":                           "Natural",
    "Agiba_beach":                           "Natural",
    "Aguilkia_Island":                       "Pharaonic",
    "Al-Qurn":                               "Natural",
    "Blue_Hole_(Red_Sea)":                   "Natural",
    "Coloured_Canyon":                       "Natural",
    "Crystal_Mountain,_White_Desert":        "Natural",
    "Gidi_Pass":                             "Natural",
    "Green_Island_(Egypt)":                  "Natural",
    "Kitchener's_Island":                    "Contemporary",
    "Lake_Timsah":                           "Natural",
    "Na'ama_Bay":                            "Natural",
    "Nabq_Protected_Area":                   "Natural",
    "Petrified_Forest_near_Maadi":           "Natural",
    "Rahmanyia_island":                      "Natural",
    "Ras_Muhammad":                          "Natural",
    "Rhoda_Island":                          "Pharaonic",
    "Soma_Bay":                              "Natural",
    "Wadi_Degla":                            "Natural",
    "Wadi_el-Raiyan":                        "Natural",
    "Wadi_el_Gemal_National_Park":           "Natural",
    # GLD-specific era overrides
    "Beni_Hasan_necropolis":                 "Pharaonic",
    "Abdin_Palace":                          "Ottoman",
    "Ras_El_Tin_Palace":                     "Ottoman",
    "Egyptian_Museum":                       "Contemporary",
    "Memphis_Museum":                        "Pharaonic",
    "Winter_Palace":                         "Contemporary",
    "Qubba_Palace":                          "Ottoman",
    "Unknown_Soldier_Memorial":              "Contemporary",
    "This_appears_to_be_related_to_Tomb_of_Rekhmire": "Pharaonic",
    "Whale_Valley":                          "Natural",
    "Red_Sea":                               "Natural",
    "Gerf_Hussein":                          "Pharaonic",
    "Kalabsha_City":                         "Pharaonic",
    "Dayr_al_Madīnah":                       "Pharaonic",
    "Kawm_Madinat_Madi":                     "Pharaonic",
    "The_First_Terrace":                     "Pharaonic",
    "Fustat":                                "Islamic",
    "Valley_of_the_nobles_Asasif":           "Pharaonic",
    "The_Great_Sphinx":                      "Pharaonic",
    "Funerary_Temple_of_Khafre":             "Pharaonic",
    "Coptic_Cairo":                          "Coptic/Byzantine",
    "National_Cultural_Centre":              "Contemporary",
    "Egyptian_Theater":                      "Contemporary",
    "Egyptian__Spice__Bazaar":               "Islamic",
    "Nilometer_in_Rhoda_Island":             "Pharaonic",
    "Gezira_Center_for_Modern_Art":          "Contemporary",
    "Garden_City,_Cairo":                    "Contemporary",
    "Gayer-Anderson_Museum":                 "Islamic",
}

# Landmarks in the GLD subset that are NOT in Egypt — exclude from merged index
NOT_IN_EGYPT = {
    "Sforza_Castle",
    "University_College_London",
    "Embassy_of_Egypt",
    "Embassy_of_the_Arab_Republic_of_Egypt",
    "Empire_House",
    "Египетский_мост",
    "This_appears_to_be_related_to_Egyptian_Embassy__Oslo",
    "This_appears_to_be_related_to_Rosicrucian_Egyptian_Museum",
    # Q&A reveals these are outside Egypt
    "Egyptian_Theater",          # DeKalb, Illinois, USA
    "Egyptian__Spice__Bazaar",   # Istanbul, Turkey
    "Albert_Al_Awal_Street",     # Unclear / not a landmark
    "Quser__Qena_Road",          # A road, not a landmark
    "Sharm_Al_Sheikh__Ras_Mohamed_Road",  # A road, not a landmark
    "Margerges_Street",          # A street, not a landmark
    "This_appears_to_be_related_to_Nile_Delta",  # Generic region, not a landmark
}

# GLD transliteration aliases: GLD name → main dataset canonical name
# Used by merge_indexes.py to deduplicate across datasets
GLD_ALIASES = {
    "Dayr_al_Madīnah":            "Deir_el-Medina",
    "Kawm_Madinat_Madi":          "Medinet_Madi",
    "Khan_el_Khalili":            "Khan_el-Khalili",
    "TT52_Nakht":                 "Tomb_of_Nakht_TT52",
    "Beni_Hasan_necropolis":      "Beni_Hassan",
    "The_Great_Sphinx":           "Great_Sphinx_of_Giza",
    "Djoser_Pyramid_complex_in_Sakkara": "Pyramid_of_Djoser",
    "El_Morsi_Abou_El_Abbas_Mosque": "Abu_el-Abbas_el-Mursi_Mosque",
    "El_Azhar_Mosque":            "Al-Azhar_Park_(Cairo)",
    "Mosque_Of_Al_Aqmar":         "Al-Aqmar_Mosque",
    "Rifaei_Mosque":              "Al-Rifa'i_Mosque",
    "Salah_El_Din_Citadel":       "Cairo_Citadel",
    "Aga_Khan_Mausoleum":         "Mausoleum_of_Aga_Khan",
    "Abdin_Palace":               "Abdeen_Palace",
    "Qubba_Palace":               "Koubbeh_Palace",
    "Ras_El_Tin_Palace":          "Ras_el-Tin_Palace",
    "Pyramid_Senusret_I":         "Pyramid_of_Senusret_I",
    "Pyramid_of_Amenemhet_I":     "Pyramid_of_Amenemhat_I",
    "Pyramid_of_Hawara":          "Pyramid_of_Amenemhat_III_in_Hawara",
    "Sesostris_II_Pyramid":       "Pyramid_of_Senusret_II",
    "Pyramid_of_Merenre_I":       "Pyramid_of_Merenre_Nemtyemsaf_I",
    "Pyramid_of_Nyuserre":        "Pyramid_of_Nyuserre_Ini",
    "Sun_Temple_of_Niuserre":     "Nyuserre_sun_temple",
    "Karnak_Temple_Complex":      "Karnak_precinct_of_Amun-Ra",
    "Dendera_Temple_Complex":     "Dendera_Temple_complex",
    "Edfu_Temple":                "Edfu_Temple",
    "Temple_of_Amenhotep_III":    "Colossi_of_Memnon",
    "Kiosk_of_Qertassi":         "Kiosk_of_Qertassi",
    "Unknown_Soldier_Memorial":   "Tomb_of_Unknown_Soldier_in_Cairo",
    "Blue_Mosque":                "Aqsunqur_Mosque",
    "Mosque_of_Amir_Al_Tunbugha_Al_Mardidini": "Mosque_of_al-Maridani",
    "Mosque_Of_Al_Nasser_Mohammad_Ibn_Qalaun": "Madrasah_of_Al-Nasir_Muhammad",
}

# ── Type classification ────────────────────────────────────────────────────────
# Each tuple: (type_label, [substring keywords — checked against lowercased folder name])
# IMPORTANT: Rules are checked top-to-bottom, first match wins.
# Longer/more-specific keywords should appear in earlier rules to avoid collisions.
TYPE_RULES = [
    ("Pyramid",       ["pyramid", "mastabat_al", "meidum",
                       "bent_pyramid", "red_pyramid",
                       "black_pyramid", "layer_pyramid", "white_pyramid",
                       "sesostris"]),
    ("Mosque",        ["mosque", "masjid", "mosque-madrassa"]),
    ("Madrassa",      ["madrasah", "madrassa"]),
    ("Temple",        ["temple", "karnak", "dendera", "edfu", "esna", "kom_ombo",
                       "ramesseum", "osireion", "speos", "deir_el-bahari",
                       "mortuary_temple", "hypostyle", "aten",
                       "kiosk_of", "new_kalabsha", "amada",
                       "qasr_qarun", "medinet_madi", "gerf_hussein",
                       "kalabsha", "funerary_temple",
                       "sun_temple"]),
    ("Museum",        ["museum"]),
    ("Library",       ["library", "bibliotheca"]),
    ("Palace",        ["palace", "heliopolis_palace"]),
    ("Church",        ["church", "cathedral", "hanging_church"]),
    ("Monastery",     ["monastery", "deir_el-muharraq", "deir_el-qadisa",
                       "paromeos", "syrian_monastery", "white_monastery",
                       "red_monastery"]),
    ("Tomb",          ["tomb", "kv17", "kv62", "wv22", "tt25", "tt52",
                       "valley_of_the_queens", "valley_of_the_nobles",
                       "valley_of_the_golden", "theban_necropolis", "dra_abu",
                       "qurnet_murai", "city_of_the_dead",
                       "umm_el-qaab", "qubbet_el-hawa", "mausoleum",
                       "fatimid_cemetery", "beni_hassan",
                       "kom_el-shoqafa", "horemheb", "rekhmire",
                       "theban_tomb"]),
    ("Monument",      ["sphinx", "colossi", "obelisk", "memorial",
                       "stele", "pompey", "pillar",
                       "nilometer", "unknown_soldier"]),
    ("Fountain",      ["sabil"]),
    ("Gate",          ["bab_al", "bab_zu"]),
    ("Fortress",      ["citadel", "fort", "qaitbay", "pharaon_island"]),
    ("Synagogue",     ["synagogue"]),
    ("Market",        ["khan_el-khalili", "khan_el_khalili", "bazaar"]),
    ("Garden/Park",   ["zoo", "aquarium", "orman",
                       "shalalat", "el-safa", "botanical",
                       "dream_park"]),
    ("Stadium",       ["stadium"]),
    ("Bridge",        ["bridge"]),
    ("Opera House",   ["opera_house"]),
    ("Dam",           ["dam"]),
    ("Canal",         ["canal", "bahr_yousef"]),
    ("Lighthouse",    ["lighthouse"]),
    ("Necropolis",    ["necropolis", "saqqara", "bagawat"]),
    ("Complex",       ["complex", "qalawun", "al-ghuri",
                       "giza_pyramid_complex", "taghri_bardi"]),
    ("Hotel",         ["hotel", "shepheard", "san_stefano", "winter_palace"]),
    ("Natural Site",  ["beach", "island", "desert", "wadi", "mountain",
                       "plateau", "canyon", "lake", "blue_hole",
                       "nabq", "ras_muhammad", "gilf", "gidi",
                       "crystal_mountain", "coloured_canyon", "petrified",
                       "sehel", "elephantine", "kitchener", "soma_bay",
                       "rhoda_island", "rahmanyia", "green_island",
                       "gebel", "siwa", "al-qurn", "agiba", "aguilkia",
                       "whale_valley", "red_sea",
                       # NOTE: "bay" removed — it collides with "bayt"
                       # Abu_Qir_Bay and Na'ama_Bay match via other rules
                       "abu_qir_bay", "na'ama_bay"]),
    ("Neighborhood",  ["mohandessin", "mokattam", "garden_city", "gezira",
                       "islamic_cairo", "corniche", "muizz_street",
                       "coptic_cairo", "fustat"]),
    ("Archaeological Site", ["abu_ghurab", "deir_el-medina"]),
    ("Street",        ["street", "road"]),
    ("Infrastructure", ["port", "pass"]),
]

# ── Era classification ────────────────────────────────────────────────────────
# IMPORTANT: Checked top-to-bottom, first match wins.
ERA_RULES = [
    ("Pharaonic",        ["pyramid", "sphinx", "mastabat", "karnak", "dendera",
                          "edfu", "esna", "ramesseum", "osireion", "deir_el-bahari",
                          "mortuary_template", "hypostyle", "kv17", "kv62", "wv22",
                          "valley_of_the", "theban", "dra_abu", "qurnet",
                          "umm_el-qaab", "gebel_el-silsila", "speos",
                          "beni_hassan", "amada", "colossi", "sun_temple",
                          "aten", "saqqara", "giza_pyramid", "great_pyramid",
                          "great_sphinx", "obelisk", "sehel",
                          "elephantine", "famine_stele", "wadi_hammamat",
                          "new_kalabsha", "island_of_bigeh", "kiosk",
                          "qubbet_el-hawa",
                          "deir_el-medina", "tomb_of_nefertari",
                          "tomb_of_kheruef", "tomb_of_nakht", "tomb_of_hetepheres",
                          "abu_ghurab", "userkaf", "nyuserre", "bent_pyramid",
                          "red_pyramid", "black_pyramid", "layer_pyramid",
                          "white_pyramid", "collections_of_the_imhotep",
                          "medinet_madi", "qasr_qarun", "meidum",
                          "sphinx_of_memphis", "abu_simbel", "derr",
                          "abydos", "luxor_temple",
                          "colossi_of_memnon", "theban_necropolis",
                          "mortuary_temple", "ramesseum",
                          "temple_of_isis", "temple_of_khonsu",
                          "temple_of_kom_ombo", "temple_of_seti",
                          "temple_of_hibis", "temple_of_derr",
                          "dendera_temple", "edfu_temple", "esna_temple",
                          "great_hypostyle", "great_temple",
                          "funerary_temple", "gerf_hussein",
                          "sesostris", "senusret", "amenemhat",
                          "tt25", "tt52", "djoser",
                          "sahure", "teti", "unas", "pepi",
                          "djedkare", "merenre", "neferefre", "neferirkare",
                          "khendjer", "khentkaus", "qakare",
                          "pyramid_of_ahmose"]),
    ("Greco-Roman",      ["greco-roman", "roman", "pompey", "kom_el-shoqafa",
                          "ptolemaic_temple"]),
    ("Coptic/Byzantine", ["church", "cathedral", "coptic", "monastery",
                          "hanging_church", "deir_el-muharraq",
                          "paromeos", "syrian_monastery", "white_monastery",
                          "red_monastery", "deir_el-qadisa"]),
    ("Ottoman",          ["muhammad_ali_mosque", "abdeen", "sakakini",
                          "baron_empain", "heliopolis_palace",
                          "koubbeh", "ras_el-tin", "sulayman"]),
    ("Islamic",          ["mosque", "masjid", "madrasah", "madrassa",
                          "al-azhar", "al-ghuri", "qalawun", "ibn_tulun",
                          "sultan_hassan", "muizz", "khan_el-khalili",
                          "khan_el_khalili",
                          "citadel", "bab_al", "bab_zu", "sabil",
                          "amir_taz", "bayt", "aqsunqur", "giyushi",
                          "attarine", "khayrbak", "emir_qurqumas",
                          "fatima_khatun", "hatem_mosque", "qaed_ibrahim",
                          "sultan_qaytbay", "sayeda_aisha", "al_fattah",
                          "al-fath", "al-nur", "al-rifa", "al-shate",
                          "al-sayeda", "al-salih", "al-aqmar", "al-ashraf",
                          "amr_ibn", "abu_haggag", "abu_el-abbas",
                          "city_of_the_dead", "fatimid", "tirbana",
                          "el-bahr", "el-maeine",
                          "taghri_bardi", "tunbugha"]),
    ("Contemporary",     ["stadium", "opera_house", "bibliotheca_alexandrina",
                          "grand_egyptian_museum", "cairo_international",
                          "dream_park", "borg_el_arab", "desouk_stadium",
                          "sohag_stadium",
                          "el_ferdan", "suez_canal_bridge", "children",
                          "san_stefano", "6_october", "hurghada_grand",
                          "port_said_lighthouse"]),
]

# ── City lookup (explicit overrides for every landmark) ───────────────────────
CITY_LOOKUP = {
    # Greater Cairo / Giza
    "6_October_Bridge": "Cairo",
    "Abdeen_Palace": "Cairo",
    "Ahmed_Shawki_Museum": "Cairo",
    "Al-Aqmar_Mosque": "Cairo",
    "Al-Ashraf_Mosque": "Cairo",
    "Al-Azhar_Park_(Cairo)": "Cairo",
    "Al-Fath_Mosque": "Cairo",
    "Al-Ghuri_Complex": "Cairo",
    "Al-Jawhara_Palace_museum": "Cairo",
    "Al-Manyal_Palace_Museum": "Cairo",
    "Al-Nur_Mosque": "Cairo",
    "Al-Rifa'i_Mosque": "Cairo",
    "Al-Salih_Tala'i_Mosque": "Cairo",
    "Al-Sayeda_Nafeesah_Mosque": "Cairo",
    "Al-Sayeda_Zainab_Mosque": "Cairo",
    "Al-Shate'e_Mosque": "Cairo",
    "Al_Fattah_Al_Alim_Mosque_(Cairo)": "Cairo",
    "Amir_Taz_Palace": "Cairo",
    "Aqsunqur_Mosque": "Cairo",
    "Bab_Zuwayla": "Cairo",
    "Bab_al-Futuh": "Cairo",
    "Bab_al-Nasr_(Cairo)": "Cairo",
    "Baron_Empain_Palace": "Cairo",
    "Bayt_Al-Suhaymi": "Cairo",
    "Ben_Ezra_Synagogue": "Cairo",
    "Cairo_Citadel": "Cairo",
    "Cairo_International_Stadium": "Cairo",
    "Cairo_Opera_House": "Cairo",
    "Children's_Civilisation_and_Creativity_Centre": "Cairo",
    "City_of_the_dead_(Cairo)": "Cairo",
    "Coptic_Museum_in_Cairo": "Cairo",
    "EL-Safa_Park": "Cairo",
    "Egyptian_Geological_Museum": "Cairo",
    "Egyptian_Museum_(Cairo)": "Cairo",
    "Egyptian_National_Library": "Cairo",
    "Egyptian_National_Military_Museum": "Cairo",
    "Emir_Qurqumas_complex": "Cairo",
    "Fatima_Khatun_Mosque": "Cairo",
    "Garden_City,_Cairo": "Cairo",
    "Gayer-Anderson_Museum": "Cairo",
    "Gezira": "Cairo",
    "Gezira_Center_for_Modern_Art": "Cairo",
    "Giyushi_Mosque,_Cairo": "Cairo",
    "Hanging_Church_(Cairo)": "Cairo",
    "Hatem_Mosque": "Cairo",
    "Heliopolis_Palace": "Cairo",
    "Islamic_Cairo": "Cairo",
    "Khan_el-Khalili": "Cairo",
    "Khayrbak_Mosque": "Cairo",
    "Koubbeh_Palace": "Cairo",
    "Madrasah_of_Al-Nasir_Muhammad": "Cairo",
    "Madrasah_of_Sarghatmish": "Cairo",
    "Mahmoud_Khalil_Museum": "Cairo",
    "Mohandessin": "Cairo",
    "Mokattam": "Cairo",
    "Mosque-Madrassa_of_Sultan_Hassan": "Cairo",
    "Mosque_of_Ibn_Tulun": "Cairo",
    "Mosque_of_Qajmas_al-Ishaqi": "Cairo",
    "Mosque_of_Qanibay_al-Muhammadi": "Cairo",
    "Mosque_of_Qanibay_al-Rammah": "Cairo",
    "Mosque_of_Sultan_Abu_al-Ila": "Cairo",
    "Mosque_of_Sultan_al-Muayyad": "Cairo",
    "Mosque_of_Sultan_al-Zahir_Baybars": "Cairo",
    "Mosque_of_al-Mahmudiya": "Cairo",
    "Mosque_of_al-Maridani": "Cairo",
    "Muhammad_Ali_Mosque": "Cairo",
    "Muizz_Street": "Cairo",
    "Museum_of_Islamic_Art,_Cairo": "Cairo",
    "Nilometer_in_Rhoda_Island": "Cairo",
    "Orman_Garden": "Cairo",
    "Petrified_Forest_near_Maadi": "Cairo",
    "Qalawun_complex": "Cairo",
    "Qasr_al-Nil_Bridge": "Cairo",
    "Relations_of_Egypt_and_Italy": "Cairo",
    "Rhoda_Island": "Cairo",
    "Sabil_of_Abd_al-Rahman_Katkhuda": "Cairo",
    "Sadat_museum": "Cairo",
    "Saint_Barbara_Church_in_Coptic_Cairo": "Cairo",
    "Saint_George_Church_in_Coptic_Cairo": "Cairo",
    "Saint_Mark's_Coptic_Orthodox_Cathedral,_Cairo": "Cairo",
    "Saint_Mark_Coptic_Orthodox_Church_(Heliopolis)": "Cairo",
    "Saints_Sergius_and_Bacchus_Church,_Cairo": "Cairo",
    "Sakakini_Palace": "Cairo",
    "Sayeda_Aisha_Mosque": "Cairo",
    "Sha'ar_Hashamayim_Synagogue_(Cairo)": "Cairo",
    "Shepheard's_Hotel": "Cairo",
    "Sulayman_Agha_al-Silahdar_Mosque": "Cairo",
    "Sulayman_Pasha_Mosque": "Cairo",
    "Sultan_Qaytbay_Complex": "Cairo",
    "Synagogue_of_Moses_Maimonides": "Cairo",
    "Tomb_of_Unknown_Soldier_in_Cairo": "Cairo",
    "Umm_Kulthum_Museum": "Cairo",
    "Wadi_Degla": "Cairo",
    "Dream_Park": "Cairo",
    "Fustat": "Cairo",
    # Giza
    "Grand_Egyptian_Museum": "Giza",
    "Giza_Plateau": "Giza",
    "Giza_Zoo": "Giza",
    "Giza_pyramid_complex": "Giza",
    "Great_Pyramid_of_Giza": "Giza",
    "Great_Sphinx_of_Giza": "Giza",
    "Pyramid_of_Khafra": "Giza",
    "Pyramid_of_Menkaure": "Giza",
    "Tomb_of_Hetepheres": "Giza",
    # Saqqara
    "Collections_of_the_Imhotep_Museum_in_Saqqara": "Saqqara",
    "Saqqara": "Saqqara",
    "Pyramid_of_Djoser": "Saqqara",
    "Pyramid_of_Teti": "Saqqara",
    "Pyramid_of_Unas": "Saqqara",
    "Pyramid_of_Userkaf": "Saqqara",
    "Pyramid_of_Pepi_I": "Saqqara",
    "Pyramid_of_Pepi_II": "Saqqara",
    "Pyramid_of_Djedkare_Isesi": "Saqqara",
    "Pyramid_of_Merenre_Nemtyemsaf_I": "Saqqara",
    "Pyramid_of_Qakare_Ibi": "Saqqara",
    "Pyramid_of_Neferefre": "Saqqara",
    "Pyramid_of_Neferirkare": "Saqqara",
    "Pyramid_of_Nyuserre_Ini": "Saqqara",
    "Pyramid_of_Khentkaus_II": "Saqqara",
    "Pyramid_of_Khendjer": "Saqqara",
    "Mastabat_al-Fir'aun": "Saqqara",
    # Abu Ghurab / Abusir
    "Abu_Ghurab": "Abu Ghurab",
    "Nyuserre_sun_temple": "Abu Ghurab",
    "Userkaf_sun_temple": "Abu Ghurab",
    "Pyramid_of_Sahure": "Abusir",
    # Memphis / Dahshur
    "Sphinx_of_Memphis": "Memphis",
    "Bent_Pyramid": "Dahshur",
    "Red_Pyramid": "Dahshur",
    "Black_Pyramid_of_Amenemhat_III": "Dahshur",
    "White_Pyramid_of_Amenemhat_II": "Dahshur",
    "Pyramid_of_Ameni_Qemau": "Dahshur",
    # Lisht / Hawara / El Lahun
    "Pyramid_of_Amenemhat_I": "Lisht",
    "Pyramid_of_Senusret_I": "Lisht",
    "Pyramid_of_Amenemhat_III_in_Hawara": "Hawara",
    "Pyramid_of_Senusret_II": "El Lahun",
    "Pyramid_of_Seila": "Seila",
    "Layer_Pyramid": "Zawiyet el-Aryan",
    "Pyramid_of_Baka": "Zawiyet el-Aryan",
    "Pyramid_of_Djedefra": "Abu Rawash",
    # Meidum / Beni Suef
    "Meidum": "Beni Suef",
    # Alexandria
    "Abu_el-Abbas_el-Mursi_Mosque": "Alexandria",
    "Abu_Qir_Bay": "Alexandria",
    "Agiba_beach": "Marsa Matrouh",
    "Alexandria_National_Museum": "Alexandria",
    "Alexandria_Opera_House": "Alexandria",
    "Alexandria_Port": "Alexandria",
    "Alexandria_Stadium": "Alexandria",
    "Alexandria_Zoo": "Alexandria",
    "Antoniadis_Palace": "Alexandria",
    "Aquarium_Grotto_Garden": "Alexandria",
    "Bibliotheca_Alexandrina": "Alexandria",
    "Bibliotheca_Alexandrina_planetarium": "Alexandria",
    "Borg_El_Arab_Stadium": "Alexandria",
    "Cathedral_of_St._Mark,_Alexandria": "Alexandria",
    "Citadel_of_Qaitbay": "Alexandria",
    "Corniche_(Alexandria)": "Alexandria",
    "Eliyahu_Hanavi_Synagogue_(Alexandria)": "Alexandria",
    "Greco-Roman_Museum,_Alexandria": "Alexandria",
    "Greek_Orthodox_Cathedral_of_Evangelismos,_Alexandria": "Alexandria",
    "Green_Island_(Egypt)": "Alexandria",
    "Kom_el-Shoqafa": "Alexandria",
    "Montaza_Palace": "Alexandria",
    "Museum_of_Constantine_P._Cavafy": "Alexandria",
    "Pompey's_Pillar,_Alexandria": "Alexandria",
    "Rahmanyia_island": "Alexandria",
    "Ras_el-Tin_Palace": "Alexandria",
    "San_Stefano_Grand_Plaza": "Alexandria",
    "Shalalat_Garden,_Alexandria": "Alexandria",
    "St._Catherine's_Cathedral,_Alexandria": "Alexandria",
    "Tirbana_mosque,_Alexandria": "Alexandria",
    # Luxor
    "Abu_Haggag_Mosque": "Luxor",
    "Al-Qurn": "Luxor",
    "Colossi_of_Memnon": "Luxor",
    "Deir_el-Bahari": "Luxor",
    "Deir_el-Medina": "Luxor",
    "Dra_Abu_el-Naga": "Luxor",
    "Great_Hypostyle_Hall_of_Karnak": "Luxor",
    "KV17": "Luxor",
    "KV62": "Luxor",
    "Karnak_precinct_of_Amun-Ra": "Luxor",
    "Luxor_Museum": "Luxor",
    "Luxor_Temple": "Luxor",
    "Mortuary_Temple_of_Hatshepsut": "Luxor",
    "Mortuary_Temple_of_Seti_I_in_Qurna": "Luxor",
    "Mortuary_Temple_of_Thutmosis_III": "Luxor",
    "Ptolemaic_Temple_of_Hathor_in_Deir_el-Medina": "Luxor",
    "Qurnet_Murai": "Luxor",
    "Ramesseum": "Luxor",
    "Temple_of_Isis_in_Deir_el-Shelwit": "Luxor",
    "Temple_of_Khonsu_in_Karnak": "Luxor",
    "Theban_Necropolis": "Luxor",
    "Tomb_of_Kheruef": "Luxor",
    "Tomb_of_Nakht_TT52": "Luxor",
    "Tomb_of_Nefertari": "Luxor",
    "Valley_of_the_Queens": "Luxor",
    "WV22": "Luxor",
    # Aswan
    "Aguilkia_Island": "Aswan",
    "Aswan_Botanical_Garden": "Aswan",
    "Aswan_High_Dam": "Aswan",
    "Aswan_Low_Dam": "Aswan",
    "Aswan_Museum": "Aswan",
    "Elephantine": "Aswan",
    "Fatimid_Cemetery_in_Aswan": "Aswan",
    "Famine_stele": "Aswan",
    "Island_of_Bigeh": "Aswan",
    "Kiosk_of_Qertassi": "Aswan",
    "Kiosk_of_Trajan_in_Philae": "Aswan",
    "Kitchener's_Island": "Aswan",
    "Mausoleum_of_Aga_Khan": "Aswan",
    "Monastery_of_Saint_Simeon_in_Aswan": "Aswan",
    "New_Kalabsha": "Aswan",
    "Nubia_Museum,_Aswan": "Aswan",
    "Qubbet_el-Hawa": "Aswan",
    "Sehel_Island": "Aswan",
    "Temple_of_Isis_in_Philae": "Aswan",
    "Unfinished_obelisk_in_Aswan": "Aswan",
    # Nubian Temples
    "Amada": "Nubia",
    "Temple_of_Derr": "Nubia",
    "Gerf_Hussein": "Nubia",
    # Upper Egypt — other cities
    "Dendera_Temple_complex": "Qena",
    "Esna_Temple": "Esna",
    "Edfu_Temple": "Edfu",
    "Gebel_el-Silsila": "Kom Ombo",
    "Temple_of_Kom_Ombo": "Kom Ombo",
    "Osireion": "Abydos",
    "Pyramid_of_Ahmose": "Abydos",
    "Temple_of_Seti_I_in_Abydos": "Abydos",
    "Umm_el-Qaab": "Abydos",
    "Mallawi_Museum": "Mallawi",
    "Beni_Hassan": "Minya",
    "Great_Temple_of_the_Aten": "Minya",
    "Speos_Artemidos": "Minya",
    "Red_Monastery": "Sohag",
    "Sohag_Stadium": "Sohag",
    "White_Monastery": "Sohag",
    # Faiyum
    "Bahr_Yousef_canal": "Faiyum",
    "Faiyum_Zoo": "Faiyum",
    "Medinet_Madi": "Faiyum",
    "Monastery_of_Saint_Samuel_the_Confessor": "Faiyum",
    "Qasr_Qarun": "Faiyum",
    "Wadi_el-Raiyan": "Faiyum",
    # Bahariya / Western Desert
    "Crystal_Mountain,_White_Desert": "Bahariya",
    "Valley_of_the_Golden_Mummies": "Bahariya",
    "Gilf_Kebir_Plateau": "Western Desert",
    "Siwa": "Siwa",
    "Gebel_el-Teir,_el-Kharga": "Kharga",
    "Temple_of_Hibis": "Kharga",
    "Bagawat": "Kharga",
    # Sinai / Red Sea
    "Blue_Hole_(Red_Sea)": "Dahab",
    "Coloured_Canyon": "Sinai",
    "Gidi_Pass": "Sinai",
    "Na'ama_Bay": "Sharm el-Sheikh",
    "Nabq_Protected_Area": "Sharm el-Sheikh",
    "Ras_Muhammad": "Sharm el-Sheikh",
    "Saint_Catherine's_Monastery,_Mount_Sinai": "Saint Catherine",
    "Sha'b_Abu_el-Nuhas": "Red Sea",
    "Pharaon_Island": "Taba",
    "Hurghada_Grand_Aquarium": "Hurghada",
    "Soma_Bay": "Hurghada",
    "Wadi_el_Gemal_National_Park": "Marsa Alam",
    # Canal Zone / Delta
    "El_Ferdan_Railway_Bridge": "Ismailia",
    "Lake_Timsah": "Ismailia",
    "Port_Said_Lighthouse": "Port Said",
    "Port_Tewfik_Memorial": "Suez",
    "Suez_Canal_Bridge": "Suez",
    "El_Alamein_Military_Museum": "El Alamein",
    # Damietta
    "Amr_Ibn_al-Aas_Mosque_(Damietta)": "Damietta",
    "El-Bahr_mosque,_Dumyat": "Damietta",
    "El-Ma'eini_mosque,_Dumyat": "Damietta",
    "Deir_el-Qadisa_Damyana": "Damietta",
    # Desouk
    "Desouk_Stadium": "Desouk",
    "Mosque_of_Saint_Ibrahim_El-Desouky": "Desouk",
    # Wadi Natrun monasteries
    "Monastery_of_Saint_Bishoy": "Wadi Natrun",
    "Monastery_of_Saint_Macarius_the_Great": "Wadi Natrun",
    "Paromeos_Monastery": "Wadi Natrun",
    "Syrian_Monastery": "Wadi Natrun",
    # Other
    "Deir_el-Muharraq": "Assiut",
    "Monastery_of_Saint_Anthony": "Red Sea",
    "Wadi_Hammamat": "Eastern Desert",
    "Attarine_Mosque": "Alexandria",
    "Qaed_Ibrahim_Mosque": "Alexandria",
    # GLD landmarks with wrong Q&A cities
    "Winter_Palace": "Luxor",
}

# ── City → Region mapping ─────────────────────────────────────────────────────
CITY_TO_REGION = {
    "Cairo": "Greater Cairo",
    "Giza": "Greater Cairo",
    "Saqqara": "Greater Cairo",
    "Memphis": "Greater Cairo",
    "Dahshur": "Greater Cairo",
    "Abu Rawash": "Greater Cairo",
    "Abu Ghurab": "Greater Cairo",
    "Abusir": "Greater Cairo",
    "Lisht": "Greater Cairo",
    "Hawara": "Middle Egypt",
    "El Lahun": "Middle Egypt",
    "Seila": "Middle Egypt",
    "Zawiyet el-Aryan": "Greater Cairo",
    "Beni Suef": "Middle Egypt",
    "Alexandria": "Alexandria",
    "Marsa Matrouh": "Alexandria",
    "Luxor": "Upper Egypt",
    "Aswan": "Upper Egypt",
    "Nubia": "Upper Egypt",
    "Qena": "Upper Egypt",
    "Esna": "Upper Egypt",
    "Edfu": "Upper Egypt",
    "Kom Ombo": "Upper Egypt",
    "Abydos": "Upper Egypt",
    "Sohag": "Upper Egypt",
    "Mallawi": "Upper Egypt",
    "Minya": "Upper Egypt",
    "Assiut": "Upper Egypt",
    "Marsa Alam": "Upper Egypt",
    "Faiyum": "Middle Egypt",
    "Bahariya": "Western Desert",
    "Kharga": "Western Desert",
    "Western Desert": "Western Desert",
    "Siwa": "Western Desert",
    "Dahab": "Sinai",
    "Sinai": "Sinai",
    "Sharm el-Sheikh": "Sinai",
    "Saint Catherine": "Sinai",
    "Taba": "Sinai",
    "Red Sea": "Red Sea Coast",
    "Hurghada": "Red Sea Coast",
    "Eastern Desert": "Red Sea Coast",
    "El Alamein": "Nile Delta",
    "Ismailia": "Canal Zone",
    "Port Said": "Canal Zone",
    "Suez": "Canal Zone",
    "Damietta": "Nile Delta",
    "Desouk": "Nile Delta",
    "Wadi Natrun": "Nile Delta",
}

# ── City → (lat, lon) centroids ───────────────────────────────────────────────
CITY_COORDS = {
    "Cairo": (30.0444, 31.2357),
    "Giza": (29.9765, 31.1313),
    "Saqqara": (29.8712, 31.2165),
    "Memphis": (29.8451, 31.2547),
    "Dahshur": (29.7911, 31.2102),
    "Abu Rawash": (30.0285, 31.0744),
    "Abu Ghurab": (29.8971, 31.1928),
    "Abusir": (29.9000, 31.2000),
    "Lisht": (29.5686, 31.2214),
    "Hawara": (29.2724, 30.9014),
    "El Lahun": (29.2334, 30.9680),
    "Seila": (29.3000, 30.9000),
    "Zawiyet el-Aryan": (29.9447, 31.1800),
    "Beni Suef": (29.0661, 31.0994),
    "Alexandria": (31.2001, 29.9187),
    "Marsa Matrouh": (31.3453, 27.2374),
    "Luxor": (25.6872, 32.6396),
    "Aswan": (24.0889, 32.8998),
    "Nubia": (22.3372, 31.6258),
    "Qena": (26.1551, 32.7160),
    "Esna": (25.2934, 32.5560),
    "Edfu": (24.9779, 32.8750),
    "Kom Ombo": (24.4524, 32.9279),
    "Abydos": (26.1855, 31.9196),
    "Sohag": (26.5590, 31.6948),
    "Mallawi": (27.7338, 30.8442),
    "Minya": (28.1099, 30.7503),
    "Assiut": (27.1808, 31.1837),
    "Faiyum": (29.3084, 30.8428),
    "Bahariya": (28.3394, 28.8680),
    "Kharga": (25.4486, 30.5559),
    "Western Desert": (26.0000, 28.0000),
    "Siwa": (29.2036, 25.5195),
    "Dahab": (28.5097, 34.5149),
    "Sinai": (29.5000, 33.5000),
    "Sharm el-Sheikh": (27.9158, 34.3300),
    "Saint Catherine": (28.5561, 33.9760),
    "Taba": (29.4964, 34.8912),
    "Red Sea": (27.2579, 33.8116),
    "Hurghada": (27.2579, 33.8116),
    "Eastern Desert": (27.0000, 33.5000),
    "Marsa Alam": (25.0656, 34.8921),
    "El Alamein": (30.8399, 28.9551),
    "Ismailia": (30.5852, 32.2654),
    "Port Said": (31.2565, 32.2841),
    "Suez": (29.9737, 32.5495),
    "Damietta": (31.4165, 31.8133),
    "Desouk": (31.1282, 30.6433),
    "Wadi Natrun": (30.3578, 30.3307),
}

# Aliases for city names found in GLD Q&A that don't match CITY_COORDS keys
CITY_ALIASES = {
    "al minya": "Minya",
    "al fekreya city": "Minya",
    "nag al hibeil": "Qena",
    "tel al amarna": "Minya",
    "sharm ash sheikh": "Sharm el-Sheikh",
    "qantara west": "Ismailia",
    "saint petersburg": "Cairo",  # Winter_Palace in Luxor, Q&A is wrong
}


def classify(name: str, rules: list) -> str:
    """Classify a landmark name using substring rules.
    Returns the first matching label, or None if no rule matches.
    """
    n = name.lower()
    for label, keywords in rules:
        for kw in keywords:
            if kw in n:
                return label
    return None


def classify_type(name: str) -> str:
    """Classify landmark type with explicit overrides checked first."""
    if name in TYPE_OVERRIDES:
        return TYPE_OVERRIDES[name]
    return classify(name, TYPE_RULES) or "Historical Site"


def classify_era(name: str) -> str:
    """Classify landmark era with explicit overrides checked first."""
    if name in ERA_OVERRIDES:
        return ERA_OVERRIDES[name]
    return classify(name, ERA_RULES) or "Islamic"


def derive_style(landmark_type: str, era: str) -> str:
    STYLE_MAP = {
        "Pharaonic": "Ancient Egyptian",
        "Greco-Roman": "Greco-Roman",
        "Coptic/Byzantine": "Coptic",
        "Ottoman": "Ottoman",
        "Contemporary": "Contemporary",
        "Islamic": "Islamic",
        "Medieval": "Medieval",
        "Prehistoric": "Prehistoric",
        "Natural": "N/A",
    }
    return STYLE_MAP.get(era, "Mixed")


def main():
    folders = sorted(
        f for f in os.listdir(IMAGES_DIR)
        if os.path.isdir(os.path.join(IMAGES_DIR, f))
    )

    rows = []
    for i, folder in enumerate(folders, start=1):
        landmark_type = classify_type(folder)
        era           = classify_era(folder)
        city          = CITY_LOOKUP.get(folder, "Cairo")
        region        = CITY_TO_REGION.get(city, "Greater Cairo")
        lat, lon      = CITY_COORDS.get(city, (30.0444, 31.2357))
        style         = derive_style(landmark_type, era)

        rows.append({
            "landmark_id":        i,
            "canonical_name":     folder,
            "landmark_type":      landmark_type,
            "historical_era":     era,
            "geographic_region":  region,
            "city":               city,
            "architectural_style": style,
            "coordinates_lat":    lat,
            "coordinates_lon":    lon,
        })

    fieldnames = [
        "landmark_id", "canonical_name", "landmark_type", "historical_era",
        "geographic_region", "city", "architectural_style",
        "coordinates_lat", "coordinates_lon",
    ]
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} landmarks -> {OUTPUT_CSV}")

    # QA summary
    from collections import Counter
    type_dist = Counter(r["landmark_type"] for r in rows)
    era_dist  = Counter(r["historical_era"] for r in rows)
    print("\nType distribution:")
    for t, n in type_dist.most_common():
        print(f"  {t:25s} {n:3d}")
    print("\nEra distribution:")
    for e, n in era_dist.most_common():
        print(f"  {e:25s} {n:3d}")


if __name__ == "__main__":
    main()
