from django.shortcuts import render,redirect
import re
from googleapiclient.discovery import build
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import boto3
import io
import base64
import torch
from transformers import BertTokenizer
import pandas as pd
from torch.utils.data import TensorDataset,DataLoader
import time
from django.http import JsonResponse
from django.contrib import messages



def videolinkandlangdetect(request):
    myapi = 'AIzaSyCT0mwFAAiIATCuEmrEMZlVwlFZ9FQIbfE'
    youtube = build('youtube', 'v3', developerKey=myapi)

    if "perform" in request.POST:
      
        link = request.POST.get('link')
        if not link:
            messages.success(request, 'Enter Video Link')  # Add your message here
            return render(request,'index.html')
    
        expression = r'(?:youtu\.be/|youtube\.com/watch\?v=)([A-Za-z0-9_-]+)'
        match = re.search(expression, link)

        if match:
            video_id = match.group(1)
        else:
            messages.success(request, 'Invalid Link') 
            return render(request,'index.html')
        
        # Retrieve all of the comments for the video
        next_page_token = None
        comments = []

        while True:
            response = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                textFormat='plainText',
                pageToken=next_page_token
            ).execute()

            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)

            next_page_token = response.get('nextPageToken')

            if not next_page_token:
                break

        # Store the comments in the session
        request.session['all_comments'] = comments
        
        english_comments = []

        

        for i in comments:
          if i.strip():
             response = translate.detect_dominant_language(Text=i)

             detected_language_code = response['Languages'][0]['LanguageCode']
             confidence_score = response['Languages'][0]['Score']

             if detected_language_code == 'en' and confidence_score > 0.99:
                 english_comments.append(i)       


        request.session['english_comments'] = english_comments
        if comments:
            
            return render(request, 'home.html', {'available': 'Your Comments are available','total_comments':len(english_comments)})
        else:
            return render(request, 'home.html', {'notavailable': 'No Comments for this video'}) 

    return render(request, 'home.html')















def viewcomments(request):

    comments = request.session.get('all_comments')
    
    if "view-comments" in request.POST:
            return render(request, 'allcommentspage.html', {'comments': comments})
    

    return render(request, 'home.html') 













def viewenglishcomments(request):

    english_comments = request.session.get('english_comments')
    
    if "english-comments" in request.POST:  
           if english_comments: 
                return render(request,'englishcommentspage.html',{'english_comments': english_comments}) 
           else: 
                return render(request,'home.html',{'english_comments_notavailable':'No English Comments'})  
  
    return render(request, 'home.html')















def piechart(request):
    english_comments = request.session.get('english_comments')
    df = pd.DataFrame(english_comments, columns=['text'])

    if 'piechartbutton' in request.POST:
        device = torch.device('cpu')
        mymodel = torch.load('D:\\Finalproject\\final\\models\\entiremodel.pt', map_location=device)
        tokenizer = BertTokenizer.from_pretrained("D:\\Finalproject\\final\\models\\token")
        mymodel.eval().to(device)

        # Tokenize input text
        encoded_inputs = tokenizer.batch_encode_plus(
        df['text'],
        add_special_tokens=True,
        return_attention_mask=True,
        max_length=200,
        return_tensors='pt',
        padding='max_length',
        truncation=True
         )

        # Extract tensors from encoded inputs
        attention_mask = encoded_inputs['attention_mask']
        input_ids = encoded_inputs['input_ids']
        dataset = TensorDataset(input_ids, attention_mask)
        loader = DataLoader(dataset, batch_size=20, shuffle=False)
        predictions = []
        with torch.no_grad():
            for batch in loader:
                input_ids, attention_masks = batch
                outputs = mymodel(input_ids, attention_mask=attention_masks)
                _, predicted = torch.max(outputs.logits, 1)
                predictions.extend(predicted.numpy())
        pos =[]
        neg = []
        neut = []

        for i in predictions:
          if i == 0:
             pos.append(i)
          elif i == 1:
             neg.append(i)
          elif i == 2:
             neut.append(i)
        # Data for the donut chart
        values = [len(pos), len(neg), len(neut)]
        labels = ["Positive", "Negative", "Neutral"]
        colors = ['green', 'red', 'yellow']

        # Filter out categories with zero values
        non_zero_values = [val for val in values if val != 0]
        non_zero_labels = [label for label, val in zip(labels, values) if val != 0]

        # Create the donut chart
        plt.figure(figsize=(4.4, 4.4))
        plt.pie(non_zero_values, labels=non_zero_labels, colors=colors, autopct="%1.1f%%")

# Create a white circle in the center (the "donut hole")
        center_circle = plt.Circle((0, 0), 0.4, color='white')
        fig = plt.gcf()
        fig.gca().add_artist(center_circle)

# Generate Base64-encoded image data for the pie chart
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)  # Rewind the buffer
        piechart = base64.b64encode(buf.read()).decode('utf-8')

        # Data for the bar chart
        sentiment = ['Positive', 'Negative', 'Neutral', 'Total']
        count = [len(pos), len(neg), len(neut), len(english_comments)]
        # Create a bar chart
        bar_width = 0.4  # Adjust this value as needed
        plt.figure(figsize=(4, 4))
        plt.bar(sentiment, count, color=['green', 'red', 'yellow', 'blue'], width=bar_width)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)

        # Generate Base64-encoded image data for the bar chart
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)  # Rewind the buffer
        barchart = base64.b64encode(buf.read()).decode('utf-8')

        return render(request, 'home.html', {'pos': len(pos), 'neg': len(neg),'neut':len(neut),
                                              'total_comments': len(english_comments),
                                              'piechart': piechart, 'barchart': barchart})
    else:
        return render(request, 'home.html',{'ok':'okmessage'})  # Or return any response you want if the button is not pressed


















def signup(request):
    if request.method == 'POST':
        return render(request, 'home.html', {'show_container1': True})
    else:
        return render(request, 'home.html', {'show_container1': False})













def login(request):
    if request.method == 'POST':
        return render(request, 'home.html', {'show_container2': True})
    else:
        return render(request, 'home.html', {'show_container2': False})    
