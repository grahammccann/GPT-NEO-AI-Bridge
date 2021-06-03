from os import system
from transformers import pipeline, GPT2Tokenizer
import json
import requests
import time
import torch

# get the model we are using EleutherAI/gpt-neo-1.3B or EleutherAI/gpt-neo-2.7B.
model_name = "EleutherAI/gpt-neo-2.7B"

# get device to use GPU or CPU.
device_to_use = "cpu"
if device_to_use == "cpu":
    device_to_use = -1
elif device_to_use == "gpu":
    device_to_use = 1	
else:
    device_to_use = -1

# get how many articles to be processed.
def count():
    req = requests.get('https://www.your-ai-writer.com/api.php?count=1')
    return req.content.strip().decode('UTF-8')
	
# get any jobs.
def jobs():
    req = requests.get('https://www.your-ai-writer.com/api.php?job=1')
    return req.content.strip().decode('UTF-8')
	
# get articles in integer to be processed.
article_count = int(count())
	
# update the db with the generated text.
def upload(article, hash, user_id):
    req = requests.get('https://www.your-ai-writer.com/api.php?insert=1&article_hash='+ hash + '&article_body=' + article + '&article_processed=yes&article_member_id=' + user_id)   

# print.
print("[+] [" + str(int(count())) + "] article(s) to be processed.")

# loop the jobs until done.
for i in range(article_count):
    
	# get the jobs AKA articles.
    job = jobs()
    j = job.split("|")
	
	# print.
    print("[+] ------------------------------------------------------------------------ [+]")	
    print("[+] Processing: " + job)
    print("[+] ------------------------------------------------------------------------ [+]")		
	 
    # tokenizers.
    gpt_tokenizer = GPT2Tokenizer.from_pretrained(model_name)  
      
    # the main AI working.	  
    generator = pipeline('text-generation', model=model_name, device=device_to_use)
    outjson = generator(j[1], do_sample=True, tokenizer=gpt_tokenizer, max_length=int(j[2]), temperature=0.8)
    outtext = json.loads(json.dumps(outjson[0]))["generated_text"]

	# print.
    print(outtext)	
	
	# upload the article to the server.
    upload(outjson[0]["generated_text"], j[0], j[3])	
	
	# print.
    print("[+] ------------------------------------------------------------------------ [+]")
    print("[+] Done! Updating hash: " + j[0])
    print("[+] ------------------------------------------------------------------------ [+]")
