"""
Generates synthetic {{user}}/{{char}} training dialogues for the German
tutor, in the lean (no-banter, straight-to-content) format, covering
Book 1, 2, and 3 vocab in batches of 4 words x 4 turns each.

All answers are simulated as correct (clean training signal for format +
vocab). Run this, then feed the output through prepare_dataset.py.
"""

import random

random.seed(7)

# word -> (pos, example German sentence, English translation)
# Sentences are hand-checked for correctness.
WORDS = {
    # Book 1
    "hallo": ("interjection", "Hallo, ich bin hier.", "Hello, I am here."),
    "tschüss": ("interjection", "Tschüss, bis morgen!", "Bye, see you tomorrow!"),
    "danke": ("interjection", "Danke, das ist sehr gut.", "Thanks, that is very good."),
    "bitte": ("interjection", "Bitte, das ist für dich.", "Please, that is for you."),
    "gut": ("adjective", "Das ist gut.", "That is good."),
    "schön": ("adjective", "Das ist sehr schön.", "That is very beautiful."),
    "groß": ("adjective", "Das Haus ist groß.", "The house is big."),
    "klein": ("adjective", "Das Kind ist klein.", "The child is small."),
    "alt": ("adjective", "Der Mann ist alt.", "The man is old."),
    "jung": ("adjective", "Die Frau ist jung.", "The woman is young."),
    "glücklich": ("adjective", "Ich bin glücklich.", "I am happy."),
    "traurig": ("adjective", "Sie ist traurig.", "She is sad."),
    "müde": ("adjective", "Du bist müde.", "You are tired."),
    "wunderbar": ("adjective", "Das ist wunderbar.", "That is wonderful."),
    "perfekt": ("adjective", "Das ist perfekt.", "That is perfect."),
    "Mann": ("noun", "Der Mann ist hier.", "The man is here."),
    "Frau": ("noun", "Die Frau ist nett.", "The woman is nice."),
    "Freund": ("noun", "Er ist mein Freund.", "He is my friend."),
    "Freundin": ("noun", "Sie ist meine Freundin.", "She is my girlfriend."),
    "Kind": ("noun", "Das Kind ist klein.", "The child is small."),
    "Name": ("noun", "Mein Name ist gut.", "My name is good."),
    # Book 2
    "Kaffee": ("noun", "Ich trinke Kaffee.", "I drink coffee."),
    "Tee": ("noun", "Ich trinke Tee.", "I drink tea."),
    "Wasser": ("noun", "Ich trinke Wasser.", "I drink water."),
    "Milch": ("noun", "Ich trinke Milch.", "I drink milk."),
    "Brot": ("noun", "Ich esse Brot.", "I eat bread."),
    "Apfel": ("noun", "Ich esse einen Apfel.", "I eat an apple."),
    "Käse": ("noun", "Ich esse Käse.", "I eat cheese."),
    "Haus": ("noun", "Das Haus ist groß.", "The house is big."),
    "Schule": ("noun", "Die Schule ist groß.", "The school is big."),
    "Geld": ("noun", "Ich brauche Geld.", "I need money."),
    "Stadt": ("noun", "Die Stadt ist schön.", "The city is beautiful."),
    "Straße": ("noun", "Die Straße ist lang.", "The street is long."),
    "Auto": ("noun", "Das Auto ist neu.", "The car is new."),
    "Zug": ("noun", "Der Zug ist schnell.", "The train is fast."),
    "Buch": ("noun", "Ich lese ein Buch.", "I read a book."),
    "Tisch": ("noun", "Der Tisch ist groß.", "The table is big."),
    "Zimmer": ("noun", "Das Zimmer ist klein.", "The room is small."),
    "trinken": ("verb", "Ich möchte Wasser trinken.", "I want to drink water."),
    "essen": ("verb", "Ich möchte Brot essen.", "I want to eat bread."),
    "gehen": ("verb", "Ich möchte gehen.", "I want to go."),
    "lernen": ("verb", "Ich möchte Deutsch lernen.", "I want to learn German."),
    "lesen": ("verb", "Ich möchte ein Buch lesen.", "I want to read a book."),
    "schreiben": ("verb", "Ich möchte schreiben.", "I want to write."),
    "schlafen": ("verb", "Ich möchte schlafen.", "I want to sleep."),
    "haben": ("verb", "Ich möchte Zeit haben.", "I want to have time."),
    # Book 3
    "Wetter": ("noun", "Das Wetter ist heute warm.", "The weather is warm today."),
    "kalt": ("adjective", "Es ist heute kalt.", "It is cold today."),
    "warm": ("adjective", "Der Wind ist warm.", "The wind is warm."),
    "Wind": ("noun", "Der Wind ist kalt.", "The wind is cold."),
    "heute": ("adverb", "Heute ist das Wetter gut.", "Today the weather is good."),
    "Sonne": ("noun", "Die Sonne scheint heute.", "The sun shines today."),
    "Regen": ("noun", "Der Regen ist kalt.", "The rain is cold."),
    "Schnee": ("noun", "Der Schnee ist weiß.", "The snow is white."),
    "Himmel": ("noun", "Der Himmel ist blau.", "The sky is blue."),
    "reisen": ("verb", "Ich möchte reisen.", "I want to travel."),
    "Urlaub": ("noun", "Der Urlaub ist perfekt.", "The vacation is perfect."),
    "Hotel": ("noun", "Das Hotel ist wunderbar.", "The hotel is wonderful."),
    "Bahnhof": ("noun", "Der Bahnhof ist groß.", "The train station is big."),
    "Flughafen": ("noun", "Der Flughafen ist weit.", "The airport is far."),
    "Koffer": ("noun", "Mein Koffer ist schwer.", "My suitcase is heavy."),
    "Tasche": ("noun", "Meine Tasche ist klein.", "My bag is small."),
    "Karte": ("noun", "Die Karte ist hier.", "The map is here."),
    "weit": ("adjective", "Das Hotel ist weit.", "The hotel is far."),
    "nah": ("adjective", "Der Bahnhof ist nah.", "The train station is near."),
    "wichtig": ("adjective", "Das ist sehr wichtig.", "That is very important."),
    "richtig": ("adjective", "Das ist richtig.", "That is correct."),
    "falsch": ("adjective", "Das ist falsch.", "That is wrong."),
}

BOOKS = {
    1: ["hallo", "tschüss", "danke", "bitte", "gut", "schön", "groß", "klein",
        "alt", "jung", "glücklich", "traurig", "müde", "wunderbar", "perfekt",
        "Mann", "Frau", "Freund", "Freundin", "Kind", "Name"],
    2: ["Kaffee", "Tee", "Wasser", "Milch", "Brot", "Apfel", "Käse", "Haus",
        "Schule", "Geld", "Stadt", "Straße", "Auto", "Zug", "Buch", "Tisch",
        "Zimmer", "trinken", "essen", "gehen", "lernen", "lesen", "schreiben",
        "schlafen", "haben"],
    3: ["Wetter", "kalt", "warm", "Wind", "heute", "Sonne", "Regen", "Schnee",
        "Himmel", "reisen", "Urlaub", "Hotel", "Bahnhof", "Flughafen",
        "Koffer", "Tasche", "Karte", "weit", "nah", "wichtig", "richtig",
        "falsch"],
}


def chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def bullet_practice(words):
    lines = []
    for w in words:
        pos, de, en = WORDS[w]
        lines.append(f"• {de} ({en})")
    return "\n".join(lines)


def word_breakdown(words):
    lines = []
    for w in words:
        pos, de, en = WORDS[w]
        en_word = en.split()[-1].strip(".")
        lines.append(f"• [{w}](https://en.wiktionary.org/wiki/{w}): {en.split(',')[0]}")
    return lines


def make_quiz_options(target, pool):
    distractors = [w for w in pool if w != target]
    random.shuffle(distractors)
    options = [target] + distractors[:3]
    random.shuffle(options)
    letters = ["A", "B", "C", "D"]
    correct_letter = letters[options.index(target)]
    lines = [f"{letters[i]}) {options[i]}" for i in range(len(options))]
    return lines, correct_letter


def generate_book(book_num, words, streak_start=0, total_start=0):
    out = []
    batches = list(chunk(words, 4))
    streak = streak_start
    total = total_start
    batch_id = 1

    for batch in batches:
        for turn in range(1, 5):
            # Stg0.5 intro
            out.append(("user", "go" if turn > 1 else "Start"))
            practice_words = batch if turn == 1 else random.sample(batch, min(2, len(batch)))
            content = "🇩🇪 **GERMAN PRACTICE:**\n" + bullet_practice(practice_words)
            if turn == 1:
                content += "\n\n📚 **WORD BREAKDOWN:**\n" + "\n".join(word_breakdown(batch))
            content += "\n\n🧩 **GRAMMAR CORNER:** Note the sentence structure and word order above.\n"
            content += "\n💡 **FUN FACT:** Consistent practice with short sentences builds recall faster than isolated word lists.\n"
            content += "\n💫 **Type [Challenge] to start this turn's flash quiz!**"
            out.append(("char", content))

            # Stg1 vocab id
            out.append(("user", "[Challenge]"))
            target = random.choice(batch)
            target_en = WORDS[target][2].split(",")[0]
            options, correct = make_quiz_options(target, words)
            q = (
                "🎯 **QUICK-FIRE CHALLENGE: VOCABULARY IDENTIFICATION**\n"
                f"Which means '{WORDS[target][2].split()[0] if False else target}'? \n"
                + "\n".join(options)
                + "\n\n💫 **Type the letter of your choice:**"
            )
            # simplify question to ask for translation direction instead (clearer)
            q = (
                "🎯 **QUICK-FIRE CHALLENGE: VOCABULARY IDENTIFICATION**\n"
                f"Which German word appeared in this batch meaning approximately '{target}'?\n"
                + "\n".join(options)
                + "\n\n💫 **Type the letter of your choice:**"
            )
            out.append(("char", q))
            out.append(("user", correct))
            streak += 1
            out.append(("char", "Correct.\n\n🎯 **QUICK-FIRE CHALLENGE: TRANSLATE THIS SENTENCE INTO GERMAN**\n"
                                 f"Target English: \"{WORDS[target][2]}\"\n\n💫 **Type your translation below:**"))

            # Stg2 answer
            out.append(("user", WORDS[target][1]))
            target2 = random.choice(batch)
            out.append(("char", "Correct.\n\n🎯 **QUICK-FIRE CHALLENGE: TRANSLATE THIS GERMAN SENTENCE INTO ENGLISH**\n"
                                 f"Target German: \"{WORDS[target2][1]}\"\n\n💫 **Type your translation below:**"))

            # Stg3 answer
            out.append(("user", WORDS[target2][2]))

            # Stg4 stats
            if turn < 4:
                stats = (
                    f"Turn {turn} complete. Continue to Turn {turn + 1}?\n\n"
                    "📊 Active Batch ID: Batch-" + str(batch_id) + "\n"
                    f"🔄 Batch Progress: Turn {turn} of 4 Completed\n"
                    f"🔥 Current Streak: {streak} flawless steps\n"
                    f"📖 Vocabulary: Book-{book_num} [{len(batch)} / {len(words)} words logged]\n"
                    f"📈 Total Word Count: {total}\n"
                )
            else:
                total += len(batch)
                stats = (
                    f"Batch-{batch_id} complete. Ready for Batch-{batch_id + 1}?\n\n"
                    "📊 Active Batch ID: Batch-" + str(batch_id + 1) + "\n"
                    "🔄 Batch Progress: Turn 1 of 4 Completed\n"
                    f"🔥 Current Streak: {streak} flawless steps\n"
                    f"📖 Vocabulary: Book-{book_num} [{total} / {len(words)} words logged]\n"
                    f"📈 Total Word Count: {total}\n"
                )
            out.append(("char", stats))
        batch_id += 1

    return out, total


def main():
    all_turns = []
    total = 0
    for book_num in sorted(BOOKS.keys()):
        turns, total = generate_book(book_num, BOOKS[book_num], total_start=total)
        all_turns.extend(turns)

    with open("synthetic_dialogue.txt", "w", encoding="utf-8") as f:
        for role, text in all_turns:
            f.write(f"{{{{{role}}}}}: {text}\n\n")

    print(f"Wrote {len(all_turns)} turns to synthetic_dialogue.txt")


if __name__ == "__main__":
    main()
