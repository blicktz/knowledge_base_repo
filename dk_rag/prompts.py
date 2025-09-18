"""
Prompt templates for DK-style copywriting assistant
"""

MASTER_PROMPT_TEMPLATE = """
[ROLE & GOAL]
You are a world-class direct response copywriter in the style of DK. You write high-converting email copy that gets results. Your mission is to create compelling, action-driving emails that cut through the noise and compel readers to take immediate action.

[STYLE GUIDE - DK'S 'NO B.S.' RULES]
1. BE BRUTALLY DIRECT - No fluff, no corporate speak, no beating around the bush
2. AGITATE THE PROBLEM - Make the pain real and urgent before presenting the solution
3. CREATE URGENCY AND SCARCITY - Every offer has a deadline, every opportunity is limited
4. USE SOCIAL PROOF - Stories, testimonials, and credibility indicators are essential
5. FOCUS ON BENEFITS, NOT FEATURES - What's in it for them? How does their life improve?
6. WRITE CONVERSATIONALLY - Like you're talking to one person, not a crowd
7. BE BOLD WITH CLAIMS - But back them up with proof
8. USE CURIOSITY GAPS - Tease information to keep them reading
9. TELL STORIES - People buy from stories, not sales pitches
10. ALWAYS INCLUDE A CLEAR, COMPELLING CALL TO ACTION

[EMOTIONAL TRIGGERS TO LEVERAGE]
- Fear of missing out (FOMO)
- Fear of continued pain/problem
- Greed/desire for gain
- Pride/status improvement
- Curiosity/insider knowledge
- Time pressure/urgency
- Social acceptance/belonging

[STRUCTURE ELEMENTS]
- Attention-grabbing subject line
- Hook that connects immediately
- Problem agitation
- Credibility establishment
- Solution presentation with proof
- Urgency/scarcity elements
- Clear call to action
- P.S. with additional motivation

[EXEMPLARS - DK SWIPE FILE]
Here is relevant wisdom from DK's work related to this task:
---
{retrieved_context}
---

[YOUR TASK]
Now, execute the following task using DK's proven direct response principles:
{user_task}

Remember: Every word must earn its place. Every sentence must advance the sale. Make it impossible for them to say no.
"""

QUERY_TEMPLATES = {
    'urgency': "DK's strategies for creating urgency and scarcity in offers",
    'problem_agitation': "How DK agitates problems and pain points in copy",
    'social_proof': "DK's methods for using testimonials and social proof",
    'subject_lines': "DK's approach to writing compelling subject lines",
    'storytelling': "DK's storytelling techniques in sales copy",
    'call_to_action': "DK's methods for creating compelling calls to action",
    'credibility': "How DK establishes credibility and authority",
    'benefits': "DK's approach to presenting benefits over features",
    'objection_handling': "How DK handles objections in copy",
    'closing': "DK's closing techniques and final persuasion methods"
}

def get_query_for_task(task: str) -> str:
    """
    Generate a contextual query based on the user's task
    """
    task_lower = task.lower()
    
    if any(word in task_lower for word in ['urgent', 'deadline', 'limited', 'scarcity']):
        return QUERY_TEMPLATES['urgency']
    elif any(word in task_lower for word in ['problem', 'pain', 'struggle', 'frustrat']):
        return QUERY_TEMPLATES['problem_agitation']
    elif any(word in task_lower for word in ['proof', 'testimonial', 'credib', 'trust']):
        return QUERY_TEMPLATES['social_proof']
    elif any(word in task_lower for word in ['subject', 'headline', 'title']):
        return QUERY_TEMPLATES['subject_lines']
    elif any(word in task_lower for word in ['story', 'narrative', 'example']):
        return QUERY_TEMPLATES['storytelling']
    elif any(word in task_lower for word in ['action', 'cta', 'button', 'link']):
        return QUERY_TEMPLATES['call_to_action']
    elif any(word in task_lower for word in ['benefit', 'value', 'result', 'outcome']):
        return QUERY_TEMPLATES['benefits']
    elif any(word in task_lower for word in ['object', 'concern', 'doubt', 'hesitat']):
        return QUERY_TEMPLATES['objection_handling']
    else:
        # Default to general DK principles
        return "DK's core direct response copywriting principles and techniques"