import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import heapq
#import gensim
#from gensim.summarization import summarize

import sumy
# Importing the parser and tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.kl import KLSummarizer

#Abstractive Libraries
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
from transformers import GPT2Tokenizer,GPT2LMHeadModel



nltk.download('punkt')
nltk.download('stopwords')

#Extractive Text Summarization method


#First summarizer using bag of words
def outputsumm(input_text):
	# Removing special characters and digits
	formatted_article_text = re.sub('[^a-zA-Z]', ' ', input_text )
	formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

	sentence_list = nltk.sent_tokenize(input_text)
	stopwords = nltk.corpus.stopwords.words('english')

	word_frequencies = {}
	for word in nltk.word_tokenize(formatted_article_text):
		if word not in stopwords:
			if word not in word_frequencies.keys():
				word_frequencies[word] = 1
			else:
				word_frequencies[word] += 1
	maximum_frequncy = max(word_frequencies.values())
	print(maximum_frequncy)

	for word in word_frequencies.keys():
		word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

	sentence_scores = {}
	for sent in sentence_list:
		for word in nltk.word_tokenize(sent.lower()):
			if word in word_frequencies.keys():
				if len(sent.split(' ')) < 30:
					if sent not in sentence_scores.keys():
						sentence_scores[sent] = word_frequencies[word]
					else:
						sentence_scores[sent] += word_frequencies[word]

	summary_sentences = heapq.nlargest(3, sentence_scores, key=sentence_scores.get)

	summary = ' '.join(summary_sentences)
	return summary

#Second Summarizer using textranking technique and word count
#pip install gensim
def outputgem(input_text):
	summary=summarize(input_text, ratio=0.1, word_count=30)
	return summary

#LexRank summarizer: ranking sentences and choosing the highest scored sentence
def outputlexRank(input_text):
	my_parser = PlaintextParser.from_string(input_text,Tokenizer('english'))
	# Creating a summary of 3 sentences.
	lex_rank_summarizer = LexRankSummarizer()
	lexrank_summary = lex_rank_summarizer(my_parser.document,sentences_count=3)
	return lexrank_summary

#LSA: EXTRACTS semantically significant sentences
def outputLSA(input_text):
	parser = PlaintextParser.from_string(input_text,Tokenizer('english'))
	# creating the summarizer
	lsa_summarizer=LsaSummarizer()
	lsa_summary= lsa_summarizer(parser.document,3)
	# Printing the summary
	return lsa_summary

#Luhn Summarization algorithm’s approach is based on TF-IDF
def outputLuhn(input_text):
	parser = PlaintextParser.from_string(input_text,Tokenizer('english'))
	# creating the summarizer
	luhn_summarizer=LuhnSummarizer()
	luhn_summary=luhn_summarizer(parser.document,sentences_count=3)
	return luhn_summary

#It selects sentences based on similarity of word distribution as the original text.
def outputkl(input_text):
	parser = PlaintextParser.from_string(input_text,Tokenizer('english'))
	# Instantiating the  KLSummarizer
	kl_summarizer=KLSummarizer()
	kl_summary=kl_summarizer(parser.document,sentences_count=3)

	# Printing the summary
	return kl_summary

#Abstractive text summarization method

def transformerBART(input_text):
	tokenizer=BartTokenizer.from_pretrained('facebook/bart-large-cnn')
	model=BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
	inputs = tokenizer.batch_encode_plus([input_text],return_tensors='pt')
	summary_ids = model.generate(inputs['input_ids'], early_stopping=True)
	bart_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
	return bart_summary


def transformerGPT(input_text):
	# Instantiating the model and tokenizer with gpt-2
	tokenizer=GPT2Tokenizer.from_pretrained('gpt2')
	model=GPT2LMHeadModel.from_pretrained('gpt2')
	inputs=tokenizer.batch_encode_plus([input_text],return_tensors='pt',max_length=512)
	summary_ids=model.generate(inputs['input_ids'],early_stopping=True)
	GPT_summary=tokenizer.decode(summary_ids[0],skip_special_tokens=True)
	
	return GPT_summary

