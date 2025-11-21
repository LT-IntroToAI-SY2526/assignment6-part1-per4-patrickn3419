# Assignment 6 Part 1 - Writeup

**Name:** Patrick Nyman  
**Date:** November 21, 2025

---

## Part 1: Understanding Your Model

### Question 1: R² Score Interpretation
What does the R² score tell you about your model? What does it mean if R² is close to 1? What if it's close to 0?

**YOUR ANSWER:**
In my model, the R2 score tells me how much of the variation in the student's scores (Scores) is explained by the hours they studied (Hours). Being clost to 1, or 100%, means that there is a strong correlation between the two variables and that there is a strong predictive power using Hours to predict Scores. If R2 is close to 0, or 0%, it means that it does not explain the variations in Scores and that many other factors are affecting the Score. Due to this, Hours would have an unsignificant relationship to Scores and thus will have weak predicting power.



---

### Question 2: Mean Squared Error (MSE)
What does the MSE (Mean Squared Error) mean in plain English? Why do you think we square the errors instead of just taking the average of the errors?

**YOUR ANSWER:**
The MSE measures the margin of error between the predicted values and the real values. It is caluculated by subtracting the y-value of the predicted value from the actual value and then squaring the difference. It is squared so negative differences are positive and to penalize larger errors.



---

### Question 3: Model Reliability
Would you trust this model to predict a score for a student who studied 10 hours? Why or why not? Consider:
- What's the maximum hours in your dataset?
- What happens when you make predictions outside the range of your training data?

**YOUR ANSWER:**
I would trust the model to predict a student who studied 10 hours, because 10 hours is not that far out of the range for the training data which goes over 9 hours. This means that the line of best fit would likely also apply to the Scores for 10 Hours. However, I would not trust the model for predicting the score of a student who studied 15 hours, because if the pattern of the actual scores change, that change would not be incorperated into the model.



---

## Part 2: Data Analysis

### Question 4: Relationship Description
Looking at your scatter plot, describe the relationship between hours studied and test scores. Is it:
- Strong or weak?
- Linear or non-linear?
- Positive or negative?

**YOUR ANSWER:**
The relationship between hours studied and test scores is very strong; they both increase according to each other. It is linear; the scores increase at a steady value. It is postiive - as the hours studied increases, the scores also increase.




---

### Question 5: Real-World Limitations
What are some real-world factors that could affect test scores that this model doesn't account for? List at least 3 factors.

**YOUR ANSWER:**
1. Hours of sleep.
2. Level of confidence.
3. Method of studying.


---

## Part 3: Code Reflection

### Question 6: Train/Test Split
Why do we split our data into training and testing sets? What would happen if we trained and tested on the same data?

**YOUR ANSWER:**
If we were to test the model on the data that we trained it on, it would be considere "cheating" because the model would not be applying paterns to new conditions. Thus, we split our data into a training and testing set, so we are able to avoid testing the model on trained data. We can also use the test data to check the amount of error.



---

### Question 7: Most Challenging Part
What was the most challenging part of this assignment for you? How did you overcome it (or what help do you still need)?

**YOUR ANSWER:**
The most challenging part of this assignment was understanding how to use the imported libraries from anaconda. I was not sure which functions were included and how to use them, but by observing the solution in the ice_cream_example.py file I got a gernal idea for function names and how to use them. 



---

## Part 4: Extending Your Learning

### Question 8: Future Applications
Describe one real-world problem you could solve with linear regression. What would be your:
- **Feature (X):** 
- **Target (Y):** 
- **Why this relationship might be linear:**

**YOUR ANSWER:**
One real world problem I could solve with linear regression would be predicting patterns in the stock market. The feature would be the time in years, and the target would be the value of the stocks. The relationship between the two on a small scale are unpredictable, but when zoomed out to a large scale, it is clear that there is a very linear pattern between the increasing value of stocks and time.



---

## Grading Checklist (for your reference)

Before submitting, make sure you have:
- [ ] Completed all functions in `a6_part1.py`
- [ ] Generated and saved `scatter_plot.png`
- [ ] Generated and saved `predictions_plot.png`
- [ ] Answered all questions in this writeup with thoughtful responses
- [ ] Pushed all files to GitHub (code, plots, and this writeup)

---

## Optional: Extra Credit (+2 points)

If you want to challenge yourself, modify your code to:
1. Try different train/test split ratios (60/40, 70/30, 90/10)
2. Record the R² score for each split
3. Explain below which split ratio worked best and why you think that is

**YOUR ANSWER:**
