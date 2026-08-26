from src.helper import extract_abcd_answer, cleanup_model_name



class Args:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model_short_name = None


# basic answers

def test_bold_answer():
    assert extract_abcd_answer("A helyes válasz a **C** opció.") == "C"

def test_bold_answer_last_wins():
    # több kiemelt betű esetén az utolsó számít
    text = "Lehetne **A** is, de végül **C**"
    assert extract_abcd_answer(text) == "C"

def test_valasz_keyword():
    assert extract_abcd_answer("A válasz: B") == "B"

def test_valasz_keyword_lowercase():
    assert extract_abcd_answer("a helyes válasz a D") == "D"

def test_bare_letter_line():
    assert extract_abcd_answer("Indoklás blabla.\n\nC") == "C"

def test_letter_line_with_colon():
    assert extract_abcd_answer("A:") == "A"

def test_letter_line_with_paren():
    assert extract_abcd_answer("B)") == "B"


# handle thinking block

def test_think_block_stripped():
    text = "<think>Maybe A? Or B? I think D is wrong.</think>\n\n**C**"
    assert extract_abcd_answer(text) == "C"

def test_think_block_multiline():
    text = "<think>\nLooking at options:\nA: wrong\nB: wrong\n</think>\nA válasz: D"
    assert extract_abcd_answer(text) == "D"

def test_unclosed_think_returns_none():
    text = "<think>Okay, let's see. Option A says... maybe B"
    assert extract_abcd_answer(text) == ""

def test_letters_inside_think_ignored():
    text = "<think>The answer is definitely A</think>\n**B**"
    assert extract_abcd_answer(text) == "B"


# priority: string patterns, last match wins

def test_bold_beats_bare_letter():
    # a szövegben van önálló "A" is, de a **C** az erősebb minta
    text = "Az A opció rossz. A helyes: **C**"
    assert extract_abcd_answer(text) == "C"


# fallback

def test_fallback_bare_letter():
    assert extract_abcd_answer("Szerintem C lesz az") == "C"

def test_fallback_last_letter_wins():
    assert extract_abcd_answer("Nem A, nem B, hanem D") == "D"


# negative examples

def test_no_answer_returns_none():
    assert extract_abcd_answer("Nem tudom megválaszolni a kérdést.") == ""

def test_empty_string():
    assert extract_abcd_answer("") == ""

def test_lowercase_letter_not_matched():
    # kisbetűs "a" névelő nem válasz
    assert extract_abcd_answer("ez a kérdés nehéz volt") == ""


# real examples

def test_real_example_from_dataset():
    output_raw = (
        "<think>\nOkay, let's tackle this question... maybe A or B. "
        "Perhaps C or D is the answer. I'll go with C.\n</think>\n\n"
        "A kérdezett adatok alapján... a legvalószínűbb válasz a **C** opció lehet, "
        "mivel a víz és a borsó egyenlő arányban keveredése stabilabb. "
        "Viszont a megadott válaszok közül a legvalószínűbb, hogy az **C**. \n\n**C**"
    )
    assert extract_abcd_answer(output_raw) == "C"

def test_valasz_then_letter_with_explanation():
    text = "A helyes válasz a D: 2 dl bor + 2 dl szóda esetén van. A helyes válasz előtti betű tehát: D."
    assert extract_abcd_answer(text) == "D"

def test_valasz_a_betu():
    assert extract_abcd_answer("A helyes válasz a betű: A") == "A"

def test_valasz_letter_in_quotes():
    assert extract_abcd_answer('A válaszok alapján a helyes válasz a "A" lehet.') == "A"

def test_valasz_betujele():
    assert extract_abcd_answer("A helyes válasz betűjele: C") == "C"


def test_cleanup_model_name_with_full_path():
    args1 = Args("/home/models/models/meta-llama-3.1-8B-instruct")
    cleanup_model_name(args1)
    assert args1.model_short_name == "meta-llama-3.1-8b-instruct"

def test_cleanup_model_name_with_repo_name():
    args2 = Args("meta-llama/Meta-Llama-3.1-8B-Instruct")
    cleanup_model_name(args2)
    assert args2.model_short_name == "meta-llama-3.1-8b-instruct"

def test_cleanup_model_name_with_simple_name():
    args3 = Args("simple-model-name")
    cleanup_model_name(args3)
    assert args3.model_short_name == "simple-model-name"

def test_cleanup_model_name_with_uppercase():
    args4 = Args("meta-llama/Llama-2-7b-chat-hf")
    cleanup_model_name(args4)
    assert args4.model_short_name == "llama-2-7b-chat-hf"