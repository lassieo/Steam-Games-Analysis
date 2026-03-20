# Steam Games Analysis

This project explores what makes a game “successful” on Steam using real-world data like user reviews and engagement patterns.

The goal is to take a messy dataset, clean it, define a meaningful success metric, and build a strong foundation for future machine learning models.

---

## Project Goal

**What actually makes a Steam game successful?**

Instead of guessing, we used data to define success in a way that reflects both:
- how much players like a game  
- and how many people have actually played/reviewed it  

---

## How We Define Success

After testing multiple approaches, we defined a game as **successful** if:

- at least **80% of its reviews are positive**, and  
- it has at least **50 total reviews**

In code:

```python
success = 1 if positive_ratio_calc >= 0.80 and review_total_calc >= 50 else 0
