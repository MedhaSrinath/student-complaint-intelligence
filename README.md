# Student Complaint Intelligence System

An NLP-based system that analyzes student complaints and automatically classifies them based on category, sentiment, emotion, and priority using text-processing and machine-learning techniques. The system also extracts important information and identifies similar or recurring complaints using semantic similarity, helping administrators prioritize issues and resolve student concerns more efficiently.

## Problem Statement

Universities receive a large number of student complaints related to academics, hostel facilities, transportation, food, infrastructure, internet services, finance, and other campus services. Since these complaints are usually written in unstructured natural language, manually reading, categorizing, prioritizing, and identifying recurring issues can be time-consuming and inconsistent.

This project aims to develop an NLP-based system that can automatically understand student complaints, classify their category, analyze their sentiment and expressed emotion, determine their priority, extract relevant information, and identify similar or recurring complaints. The system is designed to help administrators better understand common student issues and prioritize complaints that require immediate attention.

## Key Features

- **Complaint Classification**  
  Automatically categorizes complaints into relevant areas such as Academics, Hostel, IT, Infrastructure, Transport, Food, Finance, and more.

- **Sentiment Analysis**  
  Identifies whether the expressed sentiment in a complaint is Positive, Neutral, or Negative.

- **Emotion Detection**  
  Identifies the emotional tone expressed in a complaint, such as Anger, Frustration, Sadness, Fear, or Neutral.

- **Priority Detection**  
  Estimates the urgency of a complaint and classifies it into levels such as Low, Medium, High, or Critical.

- **Key Information Extraction**  
  Extracts useful information such as locations, dates, departments, and other relevant entities from complaint text.

- **Duration Extraction**
  Identifies common time spans and start-time phrases, such as “for three days” or “since Monday,” in the extracted complaint entities.

- **Similar Complaint Detection**  
  Uses semantic similarity to identify complaints that describe the same or similar issue even when different words are used.

- **Live Duplicate Suggestions**  
  Before a new complaint is submitted, TF-IDF compares it with previously submitted complaints and suggests likely matches. Students can review the existing complaint and its status before deciding whether to submit anyway. The current 20% similarity cutoff is provisional and can be evaluated with `python evaluate_duplicate_detection.py`.

- **Recurring Complaint Groups**
  Groups similar submitted complaints using TF-IDF, labels each group with its most relevant terms, and shows the number of related reports to help admins spot possible recurring issues.

- **+1 Existing Issue**
  Students can add one +1 to a recurring issue per browser session. Each +1 is counted as an additional complaint report in dashboard and report totals.

- **Recurring Issue Identification**  
  Groups similar complaints to help administrators identify problems affecting multiple students.

- **Priority Escalation**  
  Repeated reports of the same issue can be used to identify recurring high-impact problems and support escalation.

- **Resolution Suggestions** *(Planned / Optional)*  
  Based on previously handled complaints, the system may suggest basic possible resolutions for common issues.

## Example

### Input

> "The Wi-Fi in Block C has not been working for three days and nobody has responded to our complaints. This is extremely frustrating."

### Expected Analysis

```text
Category: IT / Internet
Sentiment: Negative
Emotion: Frustration
Priority: High

Location: Block C
Duration: 3 days

Similar Complaints: 12
Recurring Issue: Block C Wi-Fi