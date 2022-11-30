import sys
import os
from icecream import ic
import contextlib
with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import gpt_2_simple as gpt2
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    import logging
    logging.getLogger('tensorflow').setLevel(logging.FATAL)

import random
import pandas as pd
import numpy as np
import math
import re

from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split


class UtadaDeen:
    def __init__(
        self,
        characters=[],
        response_length=30,
        run_name="small_uafcsc_better"
    ):
        self.characters = characters
        self.sess = gpt2.start_tf_sess()
        self.response_length=response_length
        self.run_name=run_name
        self.temperature = random.randrange(65,90)/100

        self.init_classifier()


    def init_classifier(self):
        isear_df = pd.read_csv("../corpora/isear.csv", header=None)
        isear_df.columns = ["emotion", "text", ""]
        isear_df = isear_df.drop([""], axis=1)

        cleaned_text = [self.clean_text(text) for text in isear_df["text"].tolist()]

        count_vectorizer = CountVectorizer()
        training_counts = count_vectorizer.fit_transform(cleaned_text)
        bag_of_words = count_vectorizer.transform(cleaned_text)
        tfidf_transformer = TfidfTransformer()
        tfidf_transformer.fit(bag_of_words)

        self.classifier = Pipeline([
                ('vect', count_vectorizer), 
                ('tfidf', tfidf_transformer),
                ('clf', SGDClassifier(loss="log", 
                                    penalty='l1',
                                    random_state=1
                                    ))
            ])

        X_train, X_test, Y_train, Y_test = train_test_split(cleaned_text, isear_df["emotion"].tolist(), test_size=0.3, random_state=1)
        self.classifier.fit(X_train, Y_train)


    def clean_text(self, content):
        content = content.lower().strip()
        content = re.sub(r"[^a-zA-Z]", " ", str(content))
        content = re.sub(r"[\s\t\n]+", " ", content)
        tokens = [word for word in content.split() if word and word not in stopwords.words("english")]
        cleaned_text = " ".join(tokens)
        return cleaned_text


    def generate_holistic_model_response(self, conversation, character, filtered=True, random_seed=False, run_name=None):
        past_messages = conversation[::-1][-30:] #记忆三十条消息
        
        model_run = run_name if run_name else self.run_name
        # 调整seed，或许可以用grammar based？
        if "sysadmin" in model_run:
            seed = f"<|start_text|>{past_messages[-1][1].strip()}\n<|command|> "
            ic(past_messages)
            ic(1,seed)
        elif "reddit" in model_run:
            seed = "\n".join([f"<|start_text|>{character}: {sentence[1].strip()} @{sentence[0].strip()}<|end_text|>" for sentence in past_messages])
            seed += f"\n<|start_text|>{character}: "
            ic(past_messages)
            ic(2,seed)
        elif model_run.startswith("new"):
            seed = "\n".join([f"<|start_text|>{sentence[0].strip()}: {sentence[1].strip()}<|end_text|>" for sentence in past_messages])
            seed += f"\n<|start_text|>{character}: "
            ic(past_messages)
            ic(3,seed)
        else:
            seed = "\n".join([f"{sentence[0].strip()}: {sentence[1].strip()}" for sentence in past_messages])
            seed += f"\n{character}: "
            ic(past_messages)
            ic(4,seed)

        gpt2.reset_session(self.sess)
        self.sess = gpt2.start_tf_sess()
        gpt2.load_gpt2(self.sess, run_name=model_run)

        common_letters = "tiiiiiiainosriiiitainosiatwbmtikgdpr"
        response = gpt2.generate(
            self.sess,
            length=self.response_length,
            temperature=self.temperature,
            prefix=seed + random.choice(common_letters + common_letters.upper()) if random_seed else seed,
            nsamples=1,
            batch_size=1,
            run_name=model_run,
            return_as_list=True
        )[0][len(seed):].strip()

        print(f"{character} Response Length: {len(response)}\n{response}")
        truncate = re.findall(r"(.+?)(?:\<\|?end_text\|?\>)", response)
        response = truncate[0] if truncate else response
        
        return re.sub(r"<\.*?\>", "", response)


    def start_conversation(self, conversation=[], filtered=True, random_seed=False, run_name=None, character=None, characters=None):
        # 选择角色
        character = character if character else random.choice(self.characters or characters)
        response = self.generate_holistic_model_response(conversation, character, filtered=filtered, random_seed=random_seed, run_name=run_name)
        return response


if __name__ == "__main__":
    utada_deen = UtadaDeen(
        response_length=40,
        run_name="full_text_small_run1"
        # run_name="med_first_three_harry_potter"
    )

    print(utada_deen.start_conversation(conversation=[], character="harry", filtered=True))
