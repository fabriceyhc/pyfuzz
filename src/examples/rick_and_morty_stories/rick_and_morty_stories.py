# @Time    : 12/07/21 1:05 PM
# @Author  : Fabrice Harel-Canada
# @File    : rick_and_morty_stories.py

import torch
from transformers import pipeline, set_seed
from transformers.pipelines import TextGenerationPipeline

class RickAndMortyStories:
    def __init__(self, mask_bad_words=True):
        self.pipeline = pipeline("text-generation", model="e-tony/gpt2-rnm")
        if self.pipeline.tokenizer.pad_token is None:
            self.pipeline.tokenizer.pad_token = self.pipeline.tokenizer.eos_token
            self.pipeline.model.config.pad_token_id = self.pipeline.model.config.eos_token_id
        self.mask_bad_words = mask_bad_words
        self.bad_words = self.load_bad_words()

    def load_bad_words(self):
        import urllib
        bad_words = []
        try:
            file = urllib.request.urlopen(
                "https://raw.githubusercontent.com/RobertJGabriel/Google-profanity-words/master/list.txt"
            )
            for line in file:
                dline = line.decode("utf-8")
                bad_words.append(dline.split("\n")[0])
        except:
            print("Failed to load bad words list.")
        return bad_words

    def tokens2text(self, tokens):
        return self.pipeline.tokenizer.decode(tokens)

    def generate(self, inputs, max_length=250):
        outputs = self.pipeline(
                    inputs,
                    do_sample=True,
                    max_length=len(inputs) + max_length,
                    top_k=50,
                    top_p=0.95,
                    num_return_sequences=1,
                )

        output_text = self._mask_bad_words(outputs[0]["generated_text"])
        return output_text

    def _mask_bad_words(self, text):
        explicit = False

        res_text = text.lower()
        for word in self.bad_words:
            if word in res_text:
                print(word)
                res_text = res_text.replace(word, word[0] + "*" * len(word[1:]))
                explicit = True

        if explicit:
            output_text = ""
            for oword, rword in zip(text.split(" "), res_text.split(" ")):
                if oword.lower() == rword:
                    output_text += oword + " "
                else:
                    output_text += rword + " "
            text = output_text

        return text


if __name__ == "__main__":
    
    rm_story_generator = RickAndMortyStories()

    STARTERS = {
        0: "Rick: Morty, quick! Get in the car!\nMorty: Oh no, I can't do it Rick! Please not this again.\nRick: You don't have a choice! The crystal demons are going to eat you if you don't get in!",
        1: "Elon: Oh, you think you're all that Rick? Fight me in a game of space squash!\nRick: Let's go, you wanna-be genius!\nElon: SpaceX fleet, line up!",
        2: "Morty: I love Jessica, I want us to get married on Octopulon 300 and have octopus babies.\nRick: Shut up, Morty! You're not going to Octopulon 300!",
        3: "Rick: Hey there, Jerry! What a nice day for taking these anti-gravity shoes for a spin!\nJerry: Wow, Rick! You would let me try out one of your crazy gadgets?\nRick: Of course, Jerry! That's how much I respect you.",
        4: "Rick: Come on, flip the pickle, Morty. You're not gonna regret it. The payoff is huge.",
        5: "Rick: I turned myself into a pickle, Morty! Boom! Big reveal - I'm a pickle. What do you think about that? I turned myself into a pickle!",
        6: "Rick: Come on, flip the pickle, Morty. You're not gonna regret it. The payoff is huge.\nMorty: What? Where are you?\nRick: Morty, just do it! [laughing] Just flip the pickle!",
    }

    for i, starter_text in STARTERS.items():
        print("starter_text:", starter_text)
        outputs = rm_story_generator.generate(starter_text)
        texts = [out['generated_text'] for out in outputs]
        print(texts[0])
