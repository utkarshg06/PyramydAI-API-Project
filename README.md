# LavyaMidha-PyramydAI

## Overview
This project is designed to intelligently match clients with the most suitable software vendors based on a provided product category and desired capabilities. The system leverages both semantic similarity and keyword-level relevance to return highly ranked vendor recommendations via a FastAPI endpoint.

## Methodology 

### Pre-processing 
1. Parsed the Features column to extract structured data (specifically Category and description fields).
2. Cleaned both the extracted Features and the existing description using custom text preprocessing:
> Remove non-ASCII characters and hyperlinks \
> Lowercase text \
> Remove punctuation \
> Remove stopwords
3. Combined Cleaned_Features and Cleaned_Description into a unified semantic input string. This helps handle rows where the original Features field is null or sparse.

###  Keyword Extraction (KeyBERT)

Utilized KeyBERT to extract the top 20 keywords from the unified semantic input. These keywords are later used in token overlap scoring, allowing us to boost vendors that explicitly mention the user’s desired capabilities.

### Semantic Similarity with SBERT 

Used Sentence-BERT (SBERT) to compute cosine similarity between the user query and each product’s semantic input. This enables the system to capture contextual and conceptual matches beyond simple keyword overlap.

### Token Overlap

Calculated Jaccard similarity between user-provided capabilities and the vendor’s extracted keyword list. This gives additional weight to vendors that mention capabilities in exact terms.

## Final Scoring 

The final vendor ranking score is computed using a weighted formula:

Final Score = $0.6 \cdot SemanticSimilarity + 0.1 \cdot TokenOverlap + 0.3 \cdot NormalizedRating$

> Semantic similarity is the primary driver of the ranking \
> Vendor rating serves as a proxy for quality or trust \
> Token overlap offers a small bonus for explicit keyword matches \
> Vendors with a score above 0.4 are returned, and the top 10 results are presented to the user.

## Challenges 
1. Limited experience with API development — This project involved learning FastAPI from the ground up and understanding how to expose models as endpoints.
2. Inconsistent data in the Features column — Many entries were null or poorly formatted. Combining this with cleaned description data helped create meaningful semantic inputs.

## Future Steps 

1. Implement fuzzy matching or semantic expansion for software categories (e.g., if user types “crm,” also include “customer relationship management”).
2. Added more tests with a test.py file instead of short tests within the code. 



