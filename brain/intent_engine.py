"""
brain/intent_engine.py
======================
Dual-model ML intent classification for Maya.

Architecture
------------
Two independent models vote on every utterance:

  Model A — PyTorch  : BiLSTM + Attention (bag-of-words token embeddings)
  Model B — TensorFlow/Keras : 1-D CNN over character n-grams

Final prediction = argmax of averaged softmax probabilities from both.

On first run, both models are TRAINED on the built-in labelled dataset
and saved to  models/pytorch_intent.pt  and  models/tf_intent.keras.
Subsequent runs load the saved weights — inference is instant.

Automatic retrain detection
----------------------------
A sha256 fingerprint of TRAINING_DATA is saved alongside the models in
models/training_hash.txt. On every startup, _load_or_train() compares
the current fingerprint against the saved one:

  - Models missing            → train from scratch (first run)
  - Fingerprint matches       → load saved weights, instant startup
  - Fingerprint doesn't match → TRAINING_DATA changed since the models
                                 were trained — retrain automatically,
                                 no manual file deletion required

To retrain from scratch, delete the model files and restart Maya — the
missing-files path above still exists for a full manual reset. Since
the models are also several MB each, python -m brain.train_intent
remains the preferred entry point for a full manual retrain: it wipes
every saved file (including the hash) up front and prints a test-suite
classification report at the end.

Adding new intents
------------------
1. Add labelled examples to TRAINING_DATA below.
2. Restart Maya — the TRAINING_DATA fingerprint no longer matches the
   saved hash, so models retrain automatically on that startup. (Or
   run python -m brain.train_intent for the same result plus a test
   report, or delete the model files yourself — either still works.)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np

from config.settings import config

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_MODEL_DIR    = Path(__file__).parent.parent / "models"
_PT_MODEL     = _MODEL_DIR / "pytorch_intent.pt"
_TF_MODEL     = _MODEL_DIR / "tf_intent.keras"
_VOCAB_FILE   = _MODEL_DIR / "vocab.json"
_LABELS_FILE  = _MODEL_DIR / "labels.json"
# Fingerprint of TRAINING_DATA at the time the saved models were trained.
# Lets startup tell "models exist" apart from "models are stale" without
# the developer having to remember to delete files by hand — see
# _training_data_hash() / _load_or_train() below.
_HASH_FILE    = _MODEL_DIR / "training_hash.txt"

_MODEL_DIR.mkdir(exist_ok=True)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
_EMBED_DIM    = 64
_HIDDEN_DIM   = 128
_MAX_LEN      = 30        # tokens per utterance (pad / truncate)
_EPOCHS       = 40
_BATCH        = 16
_LR           = 1e-3
_CONF_THRESH  = 0.65      # below this → fall back to keyword rules

# ══════════════════════════════════════════════════════════════════════════════
# Training data  — (utterance, intent_label)
# ══════════════════════════════════════════════════════════════════════════════
TRAINING_DATA: list[tuple[str, str]] = [

    # ── datetime ──────────────────────────────────────────────────────────────
    ("what time is it",                     "get_time"),
    ("tell me the time",                    "get_time"),
    ("current time",                        "get_time"),
    ("what is the time right now",          "get_time"),
    ("what's the time",                     "get_time"),
    ("time please",                         "get_time"),
    ("clock",                               "get_time"),
    ("what time",                           "get_time"),
    ("what day is today",                   "get_date"),
    ("what is today's date",                "get_date"),
    ("tell me the date",                    "get_date"),
    ("what month is it",                    "get_date"),
    ("current date",                        "get_date"),
    ("what year is it",                     "get_date"),
    ("today's date",                        "get_date"),

    # ── reminder / timer ──────────────────────────────────────────────────────
    ("set a timer for 5 minutes",           "set_reminder"),
    ("remind me in 10 minutes",             "set_reminder"),
    ("set an alarm for 30 seconds",         "set_reminder"),
    ("timer for 2 hours",                   "set_reminder"),
    ("remind me to drink water in 15 minutes", "set_reminder"),
    ("set reminder for 1 hour",             "set_reminder"),
    ("set timer",                           "set_reminder"),
    ("alarm in 5 minutes",                  "set_reminder"),
    ("reminder after 20 minutes",           "set_reminder"),

    # ── web search ────────────────────────────────────────────────────────────
    ("search for python tutorials",         "search_web"),
    ("google machine learning",             "search_web"),
    ("look up the weather",                 "search_web"),
    ("search the web for recipes",          "search_web"),
    ("find information about black holes",  "search_web"),
    ("google latest news",                  "search_web"),
    ("search iphone 16",                    "search_web"),
    ("look up iphone",                      "search_web"),
    ("google samsung galaxy",               "search_web"),
    ("search for best laptops",             "search_web"),
    ("find news about apple",               "search_web"),
    ("look up bitcoin price",               "search_web"),
    ("search android phones",              "search_web"),

    # ── open website ─────────────────────────────────────────────────────────
    ("open youtube",                        "open_website"),
    ("go to github",                        "open_website"),
    ("open reddit",                         "open_website"),
    ("navigate to netflix",                 "open_website"),
    ("open spotify",                        "open_website"),
    ("open gmail",                          "open_website"),
    ("go to google",                        "open_website"),
    ("open twitter",                        "open_website"),
    ("go to amazon",                        "open_website"),
    ("open linkedin",                       "open_website"),
    ("navigate to stackoverflow",           "open_website"),

    # ── open app ─────────────────────────────────────────────────────────────
    ("open chrome",                         "open_app"),
    ("launch notepad",                      "open_app"),
    ("start visual studio code",            "open_app"),
    ("open file explorer",                  "open_app"),
    ("launch calculator",                   "open_app"),
    ("open word",                           "open_app"),
    ("launch excel",                        "open_app"),
    ("open task manager",                   "open_app"),
    ("start discord",                       "open_app"),
    ("open terminal",                       "open_app"),
    ("launch whatsapp",                     "open_app"),

    # ── media ─────────────────────────────────────────────────────────────────
    ("play music",                          "play_music"),
    ("play a song",                         "play_music"),
    ("start playing",                       "play_music"),
    ("pause the song",                      "pause_music"),
    ("pause music",                         "pause_music"),
    ("stop the song",                       "pause_music"),
    ("next track",                          "next_track"),
    ("skip this song",                      "next_track"),
    ("next song",                           "next_track"),
    ("previous song",                       "prev_track"),
    ("go back",                             "prev_track"),
    ("last track",                          "prev_track"),
    ("volume up",                           "volume_up"),
    ("increase the volume",                 "volume_up"),
    ("louder please",                       "volume_up"),
    ("turn it up",                          "volume_up"),
    ("volume down",                         "volume_down"),
    ("lower the volume",                    "volume_down"),
    ("quieter please",                      "volume_down"),
    ("turn it down",                        "volume_down"),
    ("mute",                                "mute"),
    ("silence the audio",                   "mute"),
    ("mute the sound",                      "mute"),

    # ── system ────────────────────────────────────────────────────────────────
    ("what is my battery level",            "system_info"),
    ("check cpu usage",                     "system_info"),
    ("how much ram am i using",             "system_info"),
    ("check disk space",                    "system_info"),
    ("battery status",                      "system_info"),
    ("cpu performance",                     "system_info"),
    ("memory usage",                        "system_info"),
    ("take a screenshot",                   "screenshot"),
    ("capture my screen",                   "screenshot"),
    ("screenshot please",                   "screenshot"),
    ("shutdown the computer",               "shutdown"),
    ("turn off my pc",                      "shutdown"),
    ("power off",                           "shutdown"),
    ("restart the system",                  "restart"),
    ("reboot my computer",                  "restart"),
    ("lock the screen",                     "lock_screen"),
    ("lock my computer",                    "lock_screen"),

    # ── weather ───────────────────────────────────────────────────────────────
    ("what's the weather",                  "get_weather"),
    ("what is the weather today",           "get_weather"),
    ("how's the weather outside",           "get_weather"),
    ("weather today",                       "get_weather"),
    ("weather forecast",                    "get_weather"),
    ("is it going to rain today",           "get_weather"),
    ("what's the temperature outside",      "get_weather"),
    ("weather in Mumbai",                   "get_weather"),
    ("weather in London",                   "get_weather"),
    ("tell me the weather",                 "get_weather"),
    ("current weather",                     "get_weather"),
    ("do I need an umbrella today",         "get_weather"),
    ("how hot is it outside",               "get_weather"),
    ("what's it like outside",              "get_weather"),
    ("weather update",                      "get_weather"),

    # ── clipboard ─────────────────────────────────────────────────────────────
    ("what's in my clipboard",              "clipboard_read"),
    ("read my clipboard",                   "clipboard_read"),
    ("what did I copy",                     "clipboard_read"),
    ("show clipboard",                      "clipboard_read"),
    ("copy hello world to clipboard",       "clipboard_write"),
    ("write this to clipboard",             "clipboard_write"),
    ("save to clipboard",                   "clipboard_write"),
    ("clear my clipboard",                  "clipboard_clear"),
    ("empty the clipboard",                 "clipboard_clear"),
    ("wipe clipboard",                      "clipboard_clear"),

    # ── timer ─────────────────────────────────────────────────────────────────
    ("set a timer for 5 minutes",           "set_timer"),
    ("set a 10 minute timer",               "set_timer"),
    ("timer for 30 seconds",                "set_timer"),
    ("start a timer",                       "set_timer"),
    ("set timer for 1 hour",                "set_timer"),
    ("2 minute timer",                      "set_timer"),
    ("remind me in 5 minutes",              "set_timer"),
    ("countdown 3 minutes",                 "set_timer"),
    ("one hour timer",                      "set_timer"),
    ("cancel the timer",                    "cancel_timer"),
    ("stop the timer",                      "cancel_timer"),
    ("cancel all timers",                   "cancel_timer"),
    ("delete timer",                        "cancel_timer"),
    ("how much time is left on the timer",  "timer_status"),
    ("timer status",                        "timer_status"),
    ("list my timers",                      "timer_status"),

    # ── notepad ───────────────────────────────────────────────────────────────
    ("take a note",                         "note_create"),
    ("note this down",                      "note_create"),
    ("write a note",                        "note_create"),
    ("save a note",                         "note_create"),
    ("create a note",                       "note_create"),
    ("note buy milk",                       "note_create"),
    ("remember this",                       "note_create"),
    ("write down call John tomorrow",       "note_create"),
    ("add to my note",                      "note_append"),
    ("append to note",                      "note_append"),
    ("add this to my notes",                "note_append"),
    ("also note that",                      "note_append"),
    ("read my notes",                       "note_read"),
    ("what are my notes",                   "note_read"),
    ("show me my notes",                    "note_read"),
    ("read the latest note",                "note_read"),
    ("list my notes",                       "note_list"),
    ("show all notes",                      "note_list"),
    ("what notes do I have",                "note_list"),
    ("delete my note",                      "note_delete"),
    ("remove that note",                    "note_delete"),
    ("open my note in notepad",             "note_open"),
    ("open the note",                       "note_open"),

    # ── greet ─────────────────────────────────────────────────────────────────
    ("hello",                               "greet"),
    ("hi maya",                             "greet"),
    ("hey there",                           "greet"),
    ("good morning",                        "greet"),
    ("good evening",                        "greet"),
    ("good afternoon",                      "greet"),
    ("hey",                                 "greet"),
    ("hi",                                  "greet"),
    ("howdy",                               "greet"),
    ("what's up",                           "greet"),
    ("yo",                                  "greet"),
    ("sup",                                 "greet"),

    # ── farewell ──────────────────────────────────────────────────────────────
    ("goodbye",                             "farewell"),
    ("bye",                                 "farewell"),
    ("see you later",                       "farewell"),
    ("go to sleep",                         "farewell"),
    ("goodnight",                           "farewell"),
    ("see ya",                              "farewell"),
    ("take care",                           "farewell"),
    ("later",                               "farewell"),
    ("exit",                                "farewell"),
    ("quit",                                "farewell"),

    # ── thanks ────────────────────────────────────────────────────────────────
    ("thank you",                           "thanks"),
    ("thanks a lot",                        "thanks"),
    ("cheers",                              "thanks"),
    ("thanks",                              "thanks"),
    ("thank you so much",                   "thanks"),
    ("appreciate it",                       "thanks"),
    ("many thanks",                         "thanks"),
    ("much appreciated",                    "thanks"),

    # ── general Q&A — wide variety of topics and phrasings ───────────────────
    ("what is photosynthesis",              "general_query"),
    ("who is elon musk",                    "general_query"),
    ("explain quantum computing",           "general_query"),
    ("how does wifi work",                  "general_query"),
    ("tell me about the moon",              "general_query"),
    ("what are black holes",                "general_query"),
    ("what is the capital of france",       "general_query"),
    ("how many planets are there",          "general_query"),
    ("what is artificial intelligence",     "general_query"),
    ("what is machine learning",            "general_query"),
    ("who invented the telephone",          "general_query"),
    ("how far is the sun",                  "general_query"),
    ("what is the speed of light",          "general_query"),
    ("what causes earthquakes",             "general_query"),
    ("what is blockchain",                  "general_query"),
    ("explain neural networks",             "general_query"),
    ("how does the internet work",          "general_query"),
    ("what is climate change",              "general_query"),
    ("what is dna",                         "general_query"),
    ("who wrote hamlet",                    "general_query"),
    ("how do vaccines work",                "general_query"),
    ("what is iphone",                      "general_query"),
    ("tell me about iphone",                "general_query"),
    ("what is android",                     "general_query"),
    ("what is samsung",                     "general_query"),
    ("tell me about tesla",                 "general_query"),
    ("what is spacex",                      "general_query"),
    ("who is steve jobs",                   "general_query"),
    ("what is apple company",               "general_query"),
    ("tell me about microsoft",             "general_query"),
    ("what is google",                      "general_query"),
    ("what is python programming",          "general_query"),
    ("explain javascript",                  "general_query"),
    ("what is a cpu",                       "general_query"),
    ("what is ram",                         "general_query"),
    ("what is the cloud",                   "general_query"),
    ("explain 5g",                          "general_query"),
    ("what is bitcoin",                     "general_query"),
    ("what is ethereum",                    "general_query"),
    ("what is nasa",                        "general_query"),
    ("who is the president of usa",         "general_query"),
    ("what is the stock market",            "general_query"),
    ("explain gravity",                     "general_query"),
    ("what is evolution",                   "general_query"),
    ("who discovered electricity",          "general_query"),
    ("what is the theory of relativity",    "general_query"),
    # ── comparison queries ────────────────────────────────────────────────────
    ("which one is better ssd or hdd",      "general_query"),
    ("which is better iphone or android",   "general_query"),
    ("which one is faster",                 "general_query"),
    ("which is the best programming language", "general_query"),
    ("which laptop should i buy",           "general_query"),
    ("compare ssd and hdd",                 "general_query"),
    ("what is the difference between ram and rom", "general_query"),
    ("python vs javascript which is better","general_query"),
    ("amd or intel which is better",        "general_query"),
    ("windows or linux which is better",    "general_query"),
    ("which is more powerful",              "general_query"),
    ("what is better mac or pc",            "general_query"),
    ("which phone is best",                 "general_query"),
    ("which one should i choose",           "general_query"),
    ("what are the differences between",    "general_query"),
    ("pros and cons of",                    "general_query"),
    ("is ssd better than hdd",              "general_query"),
    ("is python better than java",          "general_query"),
    ("tell me the difference",              "general_query"),
    ("how is x different from y",           "general_query"),

    # ── smalltalk ─────────────────────────────────────────────────────────────
    ("how are you",                         "smalltalk"),
    ("how are you doing",                   "smalltalk"),
    ("how do you feel today",               "smalltalk"),
    ("are you doing okay",                  "smalltalk"),
    ("how is it going",                     "smalltalk"),
    ("are you there",                       "smalltalk"),
    ("you still there",                     "smalltalk"),
    ("are you awake",                       "smalltalk"),
    ("talk to me",                          "smalltalk"),
    ("say something",                       "smalltalk"),
    ("i'm bored",                           "smalltalk"),
    ("entertain me",                        "smalltalk"),
    ("what are you up to",                  "smalltalk"),
    ("are you busy",                        "smalltalk"),
    ("are you listening",                   "smalltalk"),
    # informal / abbreviated
    ("how r u",                             "smalltalk"),
    ("how ru",                              "smalltalk"),
    ("hru",                                 "smalltalk"),
    ("wyd",                                 "smalltalk"),
    ("wbu",                                 "smalltalk"),
    ("u good",                              "smalltalk"),
    ("you good",                            "smalltalk"),
    ("you okay",                            "smalltalk"),
    ("u okay",                              "smalltalk"),
    ("wassup",                              "smalltalk"),
    ("wazzup",                              "smalltalk"),
    ("what u doing",                        "smalltalk"),
    ("what are you doing",                  "smalltalk"),
    ("what r u doing",                      "smalltalk"),
    ("watcha doing",                        "smalltalk"),
    ("how goes it",                         "smalltalk"),
    ("how's everything",                    "smalltalk"),
    ("how's things",                        "smalltalk"),
    ("all good",                            "smalltalk"),
    ("you there",                           "smalltalk"),
    ("are you excited",                     "smalltalk"),
    ("are you excited today",               "smalltalk"),
    ("are you happy today",                 "smalltalk"),
    ("are you sad today",                   "smalltalk"),
    ("are you tired",                       "smalltalk"),
    ("are you tired today",                 "smalltalk"),
    ("are you bored",                       "smalltalk"),
    ("are you bored today",                 "smalltalk"),
    ("are you okay today",                  "smalltalk"),
    ("are you fine",                        "smalltalk"),
    ("are you alright",                     "smalltalk"),
    ("are you ready",                       "smalltalk"),
    ("are you angry",                       "smalltalk"),
    ("are you happy",                       "smalltalk"),
    ("are you sad",                         "smalltalk"),
    ("r u tired",                           "smalltalk"),
    ("r u bored",                           "smalltalk"),
    ("r u okay",                            "smalltalk"),
    ("r u happy",                           "smalltalk"),
    ("r u sad",                             "smalltalk"),
    ("how are you feeling today",           "smalltalk"),
    ("what's your mood",                    "smalltalk"),
    ("what is your mood today",             "smalltalk"),
    ("how is your day going",               "smalltalk"),
    ("how was your day",                    "smalltalk"),
    ("are you having a good day",           "smalltalk"),
    ("you seem happy",                      "smalltalk"),
    ("you seem excited",                    "smalltalk"),
    ("you seem sad",                        "smalltalk"),
    # ── conversational follow-on reactions — these belong to smalltalk/followup
    # NOT dismissal — user is reacting to info just received, expecting a reply
    ("that means we are good now",          "smalltalk"),
    ("so we are fine then",                 "smalltalk"),
    ("that's good to know",                 "smalltalk"),
    ("sounds good to me",                   "smalltalk"),
    ("that's great",                        "smalltalk"),
    ("interesting",                         "smalltalk"),
    ("i see",                               "smalltalk"),
    ("makes sense",                         "smalltalk"),
    ("good to know",                        "smalltalk"),
    ("wow really",                          "smalltalk"),
    ("oh nice",                             "smalltalk"),
    ("that's pretty cool",                  "smalltalk"),
    ("huh okay",                            "smalltalk"),
    ("alright then",                        "smalltalk"),
    ("fair enough",                         "smalltalk"),
    ("oh i see",                            "smalltalk"),
    ("ah okay",                             "smalltalk"),
    ("so what does that mean",              "smalltalk"),
    ("what do you think about that",        "smalltalk"),
    ("that's not bad",                      "smalltalk"),
    ("that's awesome",                      "smalltalk"),
    # ── presence / arrival — "now" in personal context is NOT a datetime query ─
    ("ok but now i am here",               "smalltalk"),
    ("i am here now",                      "smalltalk"),
    ("i'm here now",                       "smalltalk"),
    ("i'm back now",                       "smalltalk"),
    ("i just got here",                    "smalltalk"),
    ("i just arrived",                     "smalltalk"),
    ("i'm back",                           "smalltalk"),
    ("i am back",                          "smalltalk"),
    ("hey i'm here",                       "smalltalk"),
    ("ok i'm here",                        "smalltalk"),
    ("now i am here",                      "smalltalk"),
    ("i just woke up",                     "smalltalk"),
    ("i'm home now",                       "smalltalk"),
    ("i got back",                         "smalltalk"),
    ("just got back",                      "smalltalk"),
    ("just got home",                      "smalltalk"),
    ("i'm at my desk",                     "smalltalk"),
    ("i'm ready now",                      "smalltalk"),
    ("i'm here and ready",                 "smalltalk"),
    ("ok i'm ready",                       "smalltalk"),
    ("here i am",                          "smalltalk"),
    ("now what",                           "smalltalk"),
    ("so now what",                        "smalltalk"),
    ("what now",                           "smalltalk"),
    ("what do we do now",                  "smalltalk"),
    ("what should i do now",               "smalltalk"),
    ("i need you now",                     "smalltalk"),
    ("i'm here maya",                      "smalltalk"),
    ("hey i just got back",                "smalltalk"),
    ("i'm online now",                     "smalltalk"),

    # ── identity ──────────────────────────────────────────────────────────────
    ("what is your name",                   "identity"),
    ("who are you",                         "identity"),
    ("tell me about yourself",              "identity"),
    ("what can you do",                     "identity"),
    ("what are your capabilities",          "identity"),
    ("are you an ai",                       "identity"),
    ("are you human",                       "identity"),
    ("are you a robot",                     "identity"),
    ("who made you",                        "identity"),
    ("who created you",                     "identity"),
    ("introduce yourself",                  "identity"),
    ("what kind of ai are you",             "identity"),

    # ── joke ──────────────────────────────────────────────────────────────────
    ("tell me a joke",                      "joke"),
    ("say something funny",                 "joke"),
    ("make me laugh",                       "joke"),
    ("do you know any jokes",               "joke"),
    ("give me a joke",                      "joke"),
    ("tell me a pun",                       "joke"),
    ("tell me a riddle",                    "joke"),
    ("funny joke please",                   "joke"),

    # ── motivate ──────────────────────────────────────────────────────────────
    ("motivate me",                         "motivate"),
    ("i need motivation",                   "motivate"),
    ("give me a quote",                     "motivate"),
    ("inspire me",                          "motivate"),
    ("say something inspiring",             "motivate"),
    ("i feel sad",                          "motivate"),
    ("i'm feeling down",                    "motivate"),
    ("cheer me up",                         "motivate"),
    ("i need encouragement",                "motivate"),
    ("say something positive",              "motivate"),

    # ── opinion ───────────────────────────────────────────────────────────────
    ("what is your favorite color",         "opinion"),
    ("do you have a favorite movie",        "opinion"),
    ("what do you think about ai",          "opinion"),
    ("do you like music",                   "opinion"),
    ("what do you prefer",                  "opinion"),
    ("do you have feelings",                "opinion"),
    ("can you think",                       "opinion"),
    ("do you dream",                        "opinion"),
    ("are you smart",                       "opinion"),
    ("are you intelligent",                 "opinion"),
    ("do you have emotions",                "opinion"),
    ("what do you think",                   "opinion"),
    ("do you like me",                      "opinion"),
    ("do you get excited",                  "opinion"),
    ("do you feel emotions",                "opinion"),
    ("can you feel happy",                  "opinion"),
    ("can you feel sad",                    "opinion"),
    ("do you actually have feelings",       "opinion"),
    ("do you enjoy talking to me",          "opinion"),
    ("do you get bored",                    "opinion"),
    ("do you get lonely",                   "opinion"),
    ("what makes you happy",                "opinion"),
    ("what makes you sad",                  "opinion"),
    ("do you have a personality",           "opinion"),
    ("are you capable of emotions",         "opinion"),

    # ── followup ──────────────────────────────────────────────────────────────
    ("tell me more",                        "followup"),
    ("can you elaborate",                   "followup"),
    ("explain further",                     "followup"),
    ("go on",                               "followup"),
    ("what else",                           "followup"),
    ("and then what",                       "followup"),
    ("give me more details",                "followup"),
    ("what do you mean",                    "followup"),
    ("can you clarify that",                "followup"),
    ("i don't understand",                  "followup"),
    ("say that again",                      "followup"),
    ("repeat that",                         "followup"),
    ("what did you say",                    "followup"),
    ("could you repeat",                    "followup"),
    ("more info",                           "followup"),
    ("continue",                            "followup"),
    ("keep going",                          "followup"),

    # ── dismissal — ONLY true cancellations/declines, not reactions ───────────
    # "that means we are good now" is NOT dismissal — it's smalltalk.
    # Dismissal = user explicitly declining, cancelling, or shutting down a topic.
    ("not right now",                       "dismissal"),
    ("no thanks",                           "dismissal"),
    ("never mind",                          "dismissal"),
    ("nevermind",                           "dismissal"),
    ("forget it",                           "dismissal"),
    ("nah",                                 "dismissal"),
    ("nope",                                "dismissal"),
    ("not now",                             "dismissal"),
    ("maybe later",                         "dismissal"),
    ("skip it",                             "dismissal"),
    ("stop",                                "dismissal"),
    ("cancel",                              "dismissal"),
    ("leave it",                            "dismissal"),
    ("not interested",                      "dismissal"),
    ("don't worry about it",               "dismissal"),
    ("it doesn't matter",                   "dismissal"),
    ("don't bother",                        "dismissal"),
    ("drop it",                             "dismissal"),
    ("ignore that",                         "dismissal"),
    ("disregard that",                      "dismissal"),
    ("actually forget it",                  "dismissal"),
    ("scratch that",                        "dismissal"),

    # ── confirm ───────────────────────────────────────────────────────────────
    ("are you sure",                        "confirm"),
    ("is that correct",                     "confirm"),
    ("really",                              "confirm"),
    ("are you certain",                     "confirm"),
    ("can you confirm that",                "confirm"),
    ("double check that",                   "confirm"),
    ("is that right",                       "confirm"),
    ("you sure about that",                 "confirm"),

    # ── perform_action — routed via a guard in _predict() before ML runs,
    # (see _ACTION_WORD_RE) so these examples are for documentation/label
    # completeness rather than the actual routing path.
    ("can you giggle",                      "perform_action"),
    ("giggle a bit",                        "perform_action"),
    ("give me a nod",                       "perform_action"),
    ("can you nod",                         "perform_action"),
    ("wink at me",                          "perform_action"),
    ("wynk at me",                          "perform_action"),
    ("can you wink",                        "perform_action"),
    ("can you wynk",                        "perform_action"),
    ("shrug for me",                        "perform_action"),
    ("can you shrug",                       "perform_action"),
    ("sigh for me",                         "perform_action"),
    ("can you sigh",                        "perform_action"),

    # ── help ──────────────────────────────────────────────────────────────────
    ("help me",                             "help"),
    ("i need help",                         "help"),
    ("what can i ask you",                  "help"),
    ("how do i use you",                    "help"),
    ("show me what you can do",             "help"),
    ("give me a list of commands",          "help"),
    ("what commands do you support",        "help"),
    ("how does this work",                  "help"),
    ("help",                                "help"),
    ("assist me",                           "help"),
]

# ── Keyword fallback (runs when ML confidence < _CONF_THRESH) ─────────────────
# ORDER MATTERS — more specific multi-word phrases must come before single words
_KEYWORD_RULES: list[tuple[str, list[str]]] = [
    # Multi-word specific phrases first
    ("search_web",    ["search for", "look up", "find information", "find news",
                       "google search"]),
    ("open_website",  ["go to", "navigate to", "open youtube", "open github",
                       "open reddit", "open netflix", "open spotify", "open gmail",
                       "open twitter", "open amazon", "open linkedin"]),
    ("set_reminder",  ["set a timer", "set a reminder", "remind me to",
                       "remind me in", "alarm for", "timer for", "reminder for"]),
    ("system_info",   ["battery level", "cpu usage", "ram usage", "disk space",
                       "battery status", "memory usage"]),
    ("lock_screen",   ["lock screen", "lock my computer", "lock the screen"]),
    ("pause_music",   ["stop music", "pause the", "stop the song"]),
    ("next_track",    ["next song", "next track", "skip this"]),
    ("prev_track",    ["previous song", "previous track", "go back to"]),
    ("volume_up",     ["volume up", "turn it up", "louder please"]),
    ("volume_down",   ["volume down", "turn it down", "quieter please"]),
    ("shutdown",      ["shut down", "turn off my", "power off"]),
    ("open_app",      ["open chrome", "open word", "open excel", "launch notepad",
                       "start discord", "open terminal", "open task manager"]),
    # Single-word / short triggers after multi-word
    ("screenshot",    ["screenshot", "capture screen"]),
    ("restart",       ["restart", "reboot"]),
    ("play_music",    ["play music", "play song", "play a song"]),
    ("mute",          ["mute", "silence"]),
    ("get_time",      ["what time", "current time", "time please"]),
    ("get_date",      ["what day", "today's date", "current date", "what month",
                       "what year"]),
    # Comparison / general query — must come BEFORE greet to prevent misclassification
    ("general_query", ["which one is", "which is better", "which is the best",
                       "which should i", "compare ", "difference between",
                       "pros and cons", "is it better", "is ssd", "is hdd",
                       "vs ", " or ", "which one"]),
    ("greet",         ["hey maya", "hi maya", "hey there", "hello maya",
                       "good morning", "good evening", "good afternoon", "howdy"]),
    ("farewell",      ["goodbye", "goodnight", "see you", "take care"]),
    # Dismissal keywords — only hard cancellation phrases, NOT reactions
    ("dismissal",     ["not right now", "no thanks", "never mind", "nevermind",
                       "forget it", "not now", "maybe later", "skip it",
                       "don't worry about it", "not interested", "leave it",
                       "scratch that", "don't bother", "drop it", "ignore that",
                       "disregard that"]),
    ("thanks",        ["thank you", "thanks", "appreciate", "much appreciated"]),
    ("help",          ["help me", "i need help", "how do i use", "assist me"]),
    # Broad single-word fallbacks LAST
    ("open_app",      ["open", "launch", "start"]),
    ("system_info",   ["battery", "cpu", "ram", "memory", "disk"]),
    ("search_web",    ["google", "search"]),
    ("get_time",      ["clock"]),
    ("farewell",      ["bye", "exit", "quit", "sleep", "later"]),
    # General query — absolute last resort
    ("general_query", ["what is", "who is", "how does", "how do", "explain",
                       "tell me about", "tell me", "what are", "why is",
                       "where is", "when did", "define", "describe"]),
]
_KEYWORD_SUFFIX = r"(?:s|es|d|ed|ing)?"
_KEYWORD_PATTERNS: list[tuple[str, re.Pattern]] = [
    (intent, re.compile(
        r"\b(?:" + "|".join(re.escape(t.strip()) for t in triggers) + r")"
        + _KEYWORD_SUFFIX + r"\b"))
    for intent, triggers in _KEYWORD_RULES
]

# ══════════════════════════════════════════════════════════════════════════════
# Automatic retrain detection
# ══════════════════════════════════════════════════════════════════════════════

def _training_data_hash() -> str:
    """
    Deterministic fingerprint of TRAINING_DATA so _load_or_train() can
    tell "saved models exist" apart from "saved models are stale" —
    completes the previously-manual retrain protocol (delete files,
    run train_intent.py) into something that also just happens
    automatically on the next startup after anyone edits TRAINING_DATA.
    """
    payload = json.dumps(TRAINING_DATA, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_saved_hash() -> Optional[str]:
    if not _HASH_FILE.exists():
        return None
    try:
        return _HASH_FILE.read_text().strip() or None
    except OSError:
        return None


def _write_saved_hash(digest: str) -> None:
    _HASH_FILE.write_text(digest)


# ══════════════════════════════════════════════════════════════════════════════
# Shared tokenizer / vocabulary
# ══════════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class _Vocab:
    PAD = 0
    UNK = 1

    def __init__(self):
        self._w2i: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}

    def build(self, corpus: list[str]) -> None:
        for text in corpus:
            for tok in _tokenize(text):
                if tok not in self._w2i:
                    self._w2i[tok] = len(self._w2i)

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self._w2i.get(t, self.UNK) for t in _tokenize(text)]
        ids = ids[:max_len] + [self.PAD] * max(0, max_len - len(ids))
        return ids

    def __len__(self) -> int:
        return len(self._w2i)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self._w2i))

    @classmethod
    def load(cls, path: Path) -> "_Vocab":
        v = cls()
        v._w2i = json.loads(path.read_text())
        return v


# ══════════════════════════════════════════════════════════════════════════════
# Model A — PyTorch BiLSTM + Attention
# ══════════════════════════════════════════════════════════════════════════════

def _build_pytorch_model(vocab_size: int, n_classes: int):
    import torch
    import torch.nn as nn

    class _Attention(nn.Module):
        def __init__(self, hidden: int):
            super().__init__()
            self.attn = nn.Linear(hidden * 2, 1)

        def forward(self, h):                           # h: (B, T, 2H)
            scores = self.attn(h).squeeze(-1)           # (B, T)
            weights = torch.softmax(scores, dim=-1)     # (B, T)
            ctx = (weights.unsqueeze(-1) * h).sum(1)    # (B, 2H)
            return ctx

    class BiLSTMIntent(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb  = nn.Embedding(vocab_size, _EMBED_DIM, padding_idx=0)
            self.lstm = nn.LSTM(_EMBED_DIM, _HIDDEN_DIM, batch_first=True,
                                bidirectional=True, num_layers=2, dropout=0.3)
            self.attn = _Attention(_HIDDEN_DIM)
            self.drop = nn.Dropout(0.4)
            self.fc   = nn.Linear(_HIDDEN_DIM * 2, n_classes)

        def forward(self, x):
            e = self.emb(x)                    # (B, T, E)
            h, _ = self.lstm(e)                # (B, T, 2H)
            ctx = self.attn(h)                 # (B, 2H)
            return self.fc(self.drop(ctx))     # (B, C)

    return BiLSTMIntent()


def _train_pytorch(X: list[list[int]], y: list[int],
                   vocab_size: int, n_classes: int,
                   save_path: Path) -> object:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = _build_pytorch_model(vocab_size, n_classes).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=_LR)
    loss_fn = nn.CrossEntropyLoss()

    Xt = torch.tensor(X, dtype=torch.long)
    yt = torch.tensor(y, dtype=torch.long)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=_BATCH, shuffle=True)

    model.train()
    for epoch in range(_EPOCHS):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            logger.debug(f"[PyTorch] epoch {epoch+1}/{_EPOCHS}  loss={total_loss/len(loader):.4f}")

    torch.save({"state": model.state_dict(),
                "vocab_size": vocab_size,
                "n_classes": n_classes}, save_path)
    logger.info(f"PyTorch model saved → {save_path}")
    return model


def _predict_pytorch(model, X_single: list[int], device) -> np.ndarray:
    import torch
    model.eval()
    with torch.no_grad():
        inp = torch.tensor([X_single], dtype=torch.long).to(device)
        logits = model(inp)
        return torch.softmax(logits, dim=-1).cpu().numpy()[0]


# ══════════════════════════════════════════════════════════════════════════════
# Model B — TensorFlow/Keras 1-D CNN
# ══════════════════════════════════════════════════════════════════════════════

def _build_tf_model(vocab_size: int, n_classes: int):
    import tensorflow as tf
    from tensorflow import keras # type: ignore

    inp  = keras.Input(shape=(_MAX_LEN,), dtype="int32")
    x    = keras.layers.Embedding(vocab_size, _EMBED_DIM, mask_zero=True)(inp)
    x    = keras.layers.Conv1D(128, 3, activation="relu", padding="same")(x)
    x    = keras.layers.Conv1D(128, 3, activation="relu", padding="same")(x)
    x    = keras.layers.GlobalMaxPooling1D()(x)
    x    = keras.layers.Dense(128, activation="relu")(x)
    x    = keras.layers.Dropout(0.4)(x)
    out  = keras.layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(_LR),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def _train_tf(X: list[list[int]], y: list[int],
              vocab_size: int, n_classes: int,
              save_path: Path) -> object:
    import numpy as np_local
    model = _build_tf_model(vocab_size, n_classes)
    Xa = np_local.array(X)
    ya = np_local.array(y)
    model.fit(Xa, ya, epochs=_EPOCHS, batch_size=_BATCH, verbose=0)
    model.save(str(save_path))
    logger.info(f"TensorFlow model saved → {save_path}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Intent Engine — public class
# ══════════════════════════════════════════════════════════════════════════════

class IntentEngine:
    """
    Dual-model ML classifier with keyword-rule fallback.

    Usage (same public API as the old rule-based engine):
        engine = IntentEngine()
        result = engine.classify("open youtube")
        # → {"intent": "open_website", "target": "youtube",
        #    "confidence": 0.93, "raw": "open youtube",
        #    "model": "ensemble"}
    """

    def __init__(self):
        self._vocab:    Optional[_Vocab] = None
        self._labels:   list[str]        = []
        self._pt_model  = None
        self._tf_model  = None
        self._pt_device = None
        self._ready     = False
        self._load_or_train()

    # ── Public ────────────────────────────────────────────────────────────────

    def classify(self, text: str) -> dict:
        text = text.strip()
        intent, confidence, source = self._predict(text)
        target = self._extract_target(text.lower(), intent)

        result = {
            "intent":     intent,
            "target":     target,
            "confidence": round(float(confidence), 3),
            "raw":        text,
            "model":      source,
        }
        logger.debug(f"Intent '{intent}' ({confidence:.2f} via {source}): '{text}'")
        return result

    # ── Initialisation ────────────────────────────────────────────────────────

    def _load_or_train(self) -> None:
        models_exist = (
            _PT_MODEL.exists() and
            _TF_MODEL.exists() and
            _VOCAB_FILE.exists() and
            _LABELS_FILE.exists()
        )

        current_hash = _training_data_hash()
        saved_hash   = _read_saved_hash()
        stale        = models_exist and saved_hash != current_hash

        if models_exist and not stale:
            logger.info("Loading saved ML intent models…")
            self._load_saved()
        else:
            if stale:
                logger.info(
                    "TRAINING_DATA has changed since the saved models were "
                    "trained — retraining automatically (no manual delete "
                    "needed; python -m brain.train_intent still works for a "
                    "full manual retrain + test run)."
                )
            else:
                logger.info("Training ML intent models (first run — please wait)…")
            self._train_and_save()
            _write_saved_hash(current_hash)

        self._ready = True
        logger.info(
            f"IntentEngine ready — {len(self._labels)} intents, "
            f"vocab={len(self._vocab)} tokens"
        )

    def _train_and_save(self) -> None:
        texts  = [t for t, _ in TRAINING_DATA]
        labels = [l for _, l in TRAINING_DATA]

        # Build vocabulary
        self._vocab = _Vocab()
        self._vocab.build(texts)
        self._vocab.save(_VOCAB_FILE)

        # Build label map
        self._labels = sorted(set(labels))
        _LABELS_FILE.write_text(json.dumps(self._labels))
        label2idx = {l: i for i, l in enumerate(self._labels)}

        X = [self._vocab.encode(t, _MAX_LEN) for t in texts]
        y = [label2idx[l] for l in labels]
        n = len(self._labels)
        v = len(self._vocab)

        # Train PyTorch
        try:
            self._pt_model = _train_pytorch(X, y, v, n, _PT_MODEL)
            import torch
            self._pt_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info("✅ PyTorch BiLSTM trained.")
        except Exception as e:
            logger.error(f"PyTorch training failed: {e}")

        # Train TensorFlow
        try:
            import os as _os
            _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
            self._tf_model = _train_tf(X, y, v, n, _TF_MODEL)
            logger.info("✅ TensorFlow CNN trained.")
        except Exception as e:
            logger.error(f"TensorFlow training failed: {e}")

    def _load_saved(self) -> None:
        self._vocab  = _Vocab.load(_VOCAB_FILE)
        self._labels = json.loads(_LABELS_FILE.read_text())

        # Load PyTorch
        try:
            import torch
            ckpt = torch.load(_PT_MODEL, map_location="cpu", weights_only=False)
            self._pt_model = _build_pytorch_model(
                ckpt["vocab_size"], ckpt["n_classes"])
            self._pt_model.load_state_dict(ckpt["state"])
            self._pt_device = torch.device("cpu")
            self._pt_model.to(self._pt_device)
            logger.debug("PyTorch model loaded.")
        except Exception as e:
            logger.warning(f"PyTorch model load failed: {e}")

        # Load TensorFlow
        try:
            import os as _os
            _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
            import tensorflow as tf
            self._tf_model = tf.keras.models.load_model(str(_TF_MODEL))
            logger.debug("TensorFlow model loaded.")
        except Exception as e:
            logger.warning(f"TensorFlow model load failed: {e}")

    # ── Prediction ────────────────────────────────────────────────────────────

    # Hard dismissal phrases — explicit cancellations only, matched EXACTLY
    # (see _predict). Conversational reactions ("that means we are good",
    # "sounds good") are NOT in this set — they should reach the LLM for a
    # context-aware reply.
    _DISMISSAL_PHRASES: frozenset = frozenset([
        "not right now", "no thanks", "never mind", "nevermind",
        "forget it", "nah", "nope", "not now", "maybe later",
        "skip it", "don't worry about it", "not interested", "leave it",
        "cancel", "stop", "no", "scratch that", "don't bother",
        "drop it", "ignore that", "disregard that",
    ])

    _DISMISSAL_TRIM_RE = re.compile(
        r"^(?:(?:ok|okay|um|uh|hmm|well|actually)\s+)*(?P<core>.+?)"
        rf"(?:\s+(?:please|{re.escape(config.name.lower())}|{re.escape(config.user_name.lower())}))*$"
    )

    @classmethod
    def _is_dismissal(cls, t: str) -> bool:
        norm = " ".join(re.sub(r"[^a-z0-9' ]+", " ", t.replace("’", "'")).split())
        m = cls._DISMISSAL_TRIM_RE.match(norm)
        return bool(m) and m.group("core") in cls._DISMISSAL_PHRASES

    # Patterns that indicate presence/arrival — these should ALWAYS go to
    # smalltalk regardless of what words like "now", "here", "back" might
    # spuriously activate in the datetime-trained ML models.
    _PRESENCE_RE = re.compile(
        r"^(ok\s+but\s+)?"
        r"(now\s+)?"
        r"(i'?m|i\s+am|i\s+just(\s+got)?|hey\s+i'?m|here\s+i)\s+"
        r"(here|back|home|ready|online|arrived|awake|at\s+my\s+desk)"
        r"|^(just\s+got\s+(back|home|here)|i\s+got\s+back|here\s+i\s+am)"
        r"|^(now\s+what|so\s+now\s+what|what\s+now|what\s+do\s+(we|i)\s+(do\s+)?now)"
        r"|(ok\s+)?(now\s+)?i'?m\s+here(\s+maya)?$",
        re.IGNORECASE,
    )

    # A direct request for one of Maya's action animations — "can you
    # giggle", "give me a nod", "wink at me", "shrug for me", "sigh a bit" —
    # should ALWAYS route to perform_action (skills/system/perform_action.py)
    # regardless of ML confidence, since the ML models have never been
    # trained on this phrasing. Matched on the bare action word (plus
    # ordinary verb conjugation) since none of these five words collide
    # with anything else in the training vocabulary — e.g. "make me laugh"
    # already maps to "joke" and is deliberately NOT included here, so the
    # existing joke behaviour is untouched.
    #
    # "wynk"/"wynks"/"wynking" is included alongside "wink" because Google
    # STT sometimes mishears it that way — same word, same action, just a
    # common ASR typo (see skills/system/perform_action.py's _ACTION_WORDS
    # for the matching fix on the skill side).
    # Questions ABOUT an action word ("what does nod mean", "why did you sigh",
    # "do you giggle") are conversation, not a request to perform it.
    _ACTION_QUESTION_RE = re.compile(
        r"\b(what|why|how|when|where|who|which)\b"
        r"|^\s*(does|do|did|is|are|was)\b"
        r"|\b(mean|means|meaning|define|definition)\b",
        re.IGNORECASE,
    )

    def _predict(self, text: str) -> tuple[str, float, str]:
        """Returns (intent, confidence, source_label)."""
        if not self._ready or self._vocab is None:
            return self._keyword_fallback(text)

        t = text.lower().strip()

        # ── Negation / dismissal guard ────────────────────────────────────────
        # Exact phrase match only (trailing punctuation ignored). Prefix
        # matching misrouted "note …", "nod …", "stop the timer" etc. —
        # those now go through the normal guards/ML.
        tokens = re.findall(r"[a-z0-9]+", t)
        if self._is_dismissal(t):
            return "dismissal", 1.0, "negation_guard"

        # ── Presence / arrival guard ──────────────────────────────────────────
        # "ok but now I am here", "I'm back", "just got home", "now what" etc.
        # These contain words like "now" / "here" that the datetime-trained ML
        # models spuriously associate with get_time / get_date. Catch them
        # before ML runs and route directly to smalltalk → LLM.
        if self._PRESENCE_RE.search(t):
            logger.debug(f"Presence guard fired for '{text}' → smalltalk")
            return "smalltalk", 1.0, "presence_guard"

        # ── Action-request guard ──────────────────────────────────────────────
        # "can you giggle", "give me a nod", "wink at me" etc. — route
        # straight to perform_action so testing/using these animations
        # doesn't depend on the ML model ever having seen this phrasing.
        if self._ACTION_WORD_RE.search(t) and not self._ACTION_QUESTION_RE.search(t):
            logger.debug(f"Action-request guard fired for '{text}' → perform_action")
            return "perform_action", 1.0, "action_guard"

        # ── Keyword-first guard ───────────────────────────────────────────────
        # Pure greetings like "hey maya" or "good evening maya" are ≤3 tokens, 
        # so they still take the keyword path
        # the guard covers short inputs only.
        # Comparison queries ("which", "vs", "or", "compare") always skip to ML.
        _COMPARISON_WORDS  = ("which", "compare", " vs ", "difference between",
                               "pros and cons", "better than", "or hdd", "or ssd")
        is_comparison = any(w in t for w in _COMPARISON_WORDS)
        use_keywords = not is_comparison and len(tokens) <= 3
        if use_keywords:
            kw_intent, kw_conf, _ = self._keyword_fallback(text)
            if kw_conf > 0:
                return kw_intent, kw_conf, "keyword_short_input"
            # No keyword match on short/greeting input → Ollama
            return "general_query", 1.0, "short_input_fallback"

        enc = self._vocab.encode(text, _MAX_LEN)
        probs_list = []

        # PyTorch prediction
        if self._pt_model is not None:
            try:
                p = _predict_pytorch(self._pt_model, enc, self._pt_device)
                probs_list.append(("pytorch", p))
            except Exception as e:
                logger.debug(f"PyTorch inference error: {e}")

        # TensorFlow prediction
        if self._tf_model is not None:
            try:
                import numpy as np_local
                inp = np_local.array([enc])
                p   = self._tf_model.predict(inp, verbose=0)[0]
                probs_list.append(("tensorflow", p))
            except Exception as e:
                logger.debug(f"TensorFlow inference error: {e}")

        if not probs_list:
            return self._keyword_fallback(text)

        # Ensemble — average probabilities
        avg_probs = np.mean([p for _, p in probs_list], axis=0)
        idx       = int(np.argmax(avg_probs))
        conf      = float(avg_probs[idx])
        intent    = self._labels[idx]
        source    = "+".join(name for name, _ in probs_list)

        if conf < _CONF_THRESH:
            kw_intent, kw_conf, _ = self._keyword_fallback(text)
            if kw_conf > 0:
                logger.debug(
                    f"ML confidence {conf:.2f} < threshold; "
                    f"using keyword fallback → '{kw_intent}'"
                )
                return kw_intent, kw_conf, "keyword_fallback"

        return intent, conf, source

    def _keyword_fallback(self, text: str) -> tuple[str, float, str]:
        t = text.lower()
        for intent, pattern in _KEYWORD_PATTERNS:
            if pattern.search(t):
                return intent, 1.0, "keyword"
        return "unknown", 0.0, "keyword"

    # ── Target extraction ─────────────────────────────────────────────────────

    def _extract_target(self, text: str, intent: str) -> str:
        """Strip the intent trigger to leave the target entity."""
        trigger_map = {
            "open_app":     ["open", "launch", "start"],
            "search_web":   ["search for", "google", "look up", "search"],
            "open_website": ["open website", "go to", "navigate to", "open"],
            "set_reminder": ["remind me to", "remind me", "set a timer for",
                             "set a reminder for", "timer for", "alarm for"],
        }
        trigger_hit = False
        for trigger in trigger_map.get(intent, []):
            if trigger in text:
                trigger_hit = True
                after = text.split(trigger, 1)[-1].strip()
                after = re.sub(r"^(for|to|the|a|an|me)\s+", "", after)
                if after:
                    return after
        # A trigger matched but nothing followed it ("search", "open"): return ""
        # so the skill's own clarification prompt fires instead of echoing the
        # trigger word back as the target. No trigger matched at all (ML-routed
        # phrasing) keeps the old whole-utterance fallback.
        return "" if trigger_hit else text