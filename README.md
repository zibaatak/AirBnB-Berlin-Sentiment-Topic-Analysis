# AirBnB-Sentiment Analysis and Topic Modeling

## Table of Contents
- [Overview](#overview)
- [Data](#data)
- [Results](#results)
- [Exploratory Data Analysis](#ExploratoryDataAnalysis)
- [Summary](#summary)
- [Acknowledgments](#acknowledgments)

## Overview


## Data
The analysis begins with the examination of two primary datasets: Reviews and Neighbourhoods GeoJSON. The Reviews dataset contains 315,479 entries, each representing a guest review. This dataset includes essential information such as listing_id, date, reviewer_id, reviewer_name, comments, and the detected_language of the review.

The Neighbourhoods GeoJSON dataset provides information about Berlin's neighborhoods, including geographical boundaries. It contains 140 entries and features like neighborhood, neighbourhood_group, and geometry.

## Results
**Data Preprocessing**
- Handling Missing Values:
The Reviews dataset exhibits no missing values across its columns, indicating that the dataset is complete and ready for analysis.

- Selecting English Reviews:
The analysis focuses on English reviews, which comprise a substantial portion of the dataset. This selection allows for effective sentiment analysis and topic modeling using natural language processing (NLP) techniques.

- Data Cleaning:
Text preprocessing was applied to the selected English reviews, which involved removing special characters, lemmatization, and other techniques to enhance the quality of the text data for analysis.

**Topic Modeling**
**Latent Dirichlet Allocation (LDA)**
LDA was employed to uncover latent topics within the English reviews. The analysis yielded multiple topic models with varying numbers of topics. Here are the primary topics extracted from the reviews:

With 3 Topics more cohesive topic allocation was possible:

- Convenience and Location: As in the previous model, this topic emphasizes the convenience and location aspects of the accommodations.
- Quality Accommodations: This topic continues to focus on the quality of accommodations, including attributes like cleanliness and amenities.
- Positive Guest Experiences: This topic highlights the positive experiences of guests, showcasing their enjoyment and intent to return.

**Sentiment Analysis**
**VADER Sentiment Analysis**
The VADER (Valence Aware Dictionary and Sentiment Reasoner) sentiment analysis tool was employed to assess the sentiment of the reviews. The sentiment analysis results provided a compound score for each review, which was categorized into three sentiment labels:

- Positive: Reviews with a compound score greater than or equal to 0.05.
- Neutral: Reviews with a compound score between -0.05 and 0.05.
- Negative: Reviews with a compound score less than or equal to -0.05.

The sentiment analysis results revealed the following distribution of sentiment labels:
- Positive: 303,693 reviews
- Neutral: 6,421 reviews
- Negative: 5,365 reviews

**Key Findings**
- Topic Insights: The LDA topic modeling unveiled distinct themes within the reviews, emphasizing the significance of convenience, quality accommodations, and positive guest experiences. This information can guide property owners and hosts in understanding guest priorities.

- Sentiment Analysis: A significant majority of reviews expressed positive sentiments (approximately 96% of reviews). This suggests a generally favorable guest experience across Airbnb listings in Berlin.

In the link below you can find a sentiment-based map of the listings.
https://public.tableau.com/views/AirBnbBerlinSentimentMap/Dashboard1?:language=en-US&:display_count=n&:origin=viz_share_link





