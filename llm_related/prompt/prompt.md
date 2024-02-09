
# input control
-  json input
```
Given a part of video subtitles JSON array as shown below:
```

# planning
- step by step planning
```
Your job is trying to generate the subtitles' outline with follow steps:

1. Extract an useful information as the outline context,
2. exclude out-of-context parts and irrelevant parts,
3. exclude text like "[Music]", "[Applause]", "[Laughter]" and so on,
4. summarize the useful information to one-word as the outline title.
```

# output control
- no explanation
```
Do not output any redundant explanation.
```
- return json
```
Return a JSON array as shown below:

```json
[
  {{
    "<field>": <type> field, <description>
  }}
]
\n```
```
