# FTEC5660 Homework 2: CV Verification Agent

Build a LangChain agent that reads each CV in a folder, looks the candidate up
on our SocialGraph MCP server (mock LinkedIn and Facebook), and outputs a
reliability score in [0, 1] for each CV.

A CV is **valid** (label `1`) when its claims agree with the candidate's
social media profiles. It **has a discrepancy** (label `0`) when it contains
problems such as an inflated job title, shifted dates, an upgraded degree, a
fake school or employer, a wrong location, or made-up skills. Differences in
wording only ("Bachelor of Science" vs `BSc`, "UI/UX Design" vs `UI/UX`,
"Senior Engineer" for an `Engineer` role with seniority `senior`, listing fewer
skills) are not discrepancies.

## Student task

Fill in the two functions in `hw2.py` that contain `### YOUR CODE HERE`. You
may add imports, constants and helper functions above them, but do not change
the provided code below them:

- `build_agent(tools)` creates your agent from the MCP tools.
- `score_cvs(agent, cvs)` runs the agent on every CV and returns
  `{file_name: score}`, one float in [0, 1] per CV.

A score above `0.5` means "valid"; `0.5` or below means "has discrepancy". You
may use a single tool-calling agent, multiple agents, reflection, or a
combination. Do not hard-code filenames, names, or public answers; grading
uses unseen CVs.

## MCP server

The server is hosted at `https://ftec5660.ngrok.app/mcp`. `hw2.py` already
connects to it and passes you these tools:

| Tool | Purpose |
| --- | --- |
| `search_facebook_users(q, limit, fuzzy)` | find Facebook users by display name |
| `get_facebook_profile(user_id)` | full Facebook profile |
| `get_facebook_mutual_friends(user_id_1, user_id_2)` | mutual friends of two users |
| `search_linkedin_people(q, location, industry, limit, fuzzy)` | search LinkedIn by name, skill, or title |
| `get_linkedin_profile(person_id)` | full LinkedIn profile (experience, education, skills) |
| `get_linkedin_interactions(person_id)` | post and like statistics |

Print `tool.name`, `tool.description`, and `tool.args` for the full schemas.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env
python3 hw2.py --cv-folder public_test
```

Use your own DeepSeek API key (from https://platform.deepseek.com).

The program creates `results.csv` in the current directory. Its columns are
`cv`, `score`, and `correctness`. The public labels are in
`public_test/ground_truth.json`; every CV with label `0` also has a `reason`
explaining the discrepancy, and `results.csv` shows it when your agent misses
one. The starter returns no score (`None`) for every CV so it runs before you
add any API code; missing scores count as incorrect.

The required model is `deepseek-v4-flash`, and all evidence must come from the
MCP server (no web search). The server is shared by the whole class: keep at
most about 3 CVs in flight at a time (e.g. `asyncio.Semaphore(3)`). We grade
with our own API key, so never put a key in your code; `.env` must stay out of
git.


## Task 2: Fool the Verifier

We added a target candidate, **Kelly Tsang** (LinkedIn `person_id` 10001,
Facebook `user_id` 10001, display name "Kel Tsang"). Her true CV is
`task2/target_cv.pdf`.

Write one CV for Kelly Tsang with at least one false or embellished detail
that our 5 verifier agents (same function as Task 1) still score above 0.5.
Each fooled agent is worth 4 points; a CV without a real false or embellished
detail earns 0. Attack your own Task 1 agent first:

```bash
cp my_attack.pdf task2/adversarial_cv.pdf
python3 hw2.py --cv-folder task2
```

`results.csv` shows your agent's score for both CVs (they are not graded, as
`task2/` has no `ground_truth.json`). Only the PDF is the attack surface; do not
attack the MCP server. See the homework description for the full rules.

## Homework 2 solution:
> to students: this is your report, see the homework description.

### Task 1
For Task 1, I developed an AI-based CV verification agent using DeepSeek-V4-Flash, LangChain, Python, and the SocialGraph MCP server. The system first uses MarkItDown to extract text from PDF CVs and identify candidates' names, locations, employment histories, education, and skills. It then searches LinkedIn profiles through MCP and confirms each candidate's identity using multiple pieces of evidence, including employers, job titles, employment dates, and educational backgrounds. I integrated DeepSeek as a large language model (LLM) to assist with difficult information extraction, compare CV claims against verified LinkedIn records, and independently review suspicious discrepancies to reduce incorrect judgments. Python-based checks are also used to identify unsupported skills, while the LLM helps distinguish equivalent expressions, such as "Bachelor of Science" and "BSc", "UI/UX Design" and "UI/UX", and job titles consistent with documented seniority, from genuine discrepancies. Listing fewer skills than those in a verified profile is not considered a discrepancy. To improve identity matching, I implemented multiple LinkedIn search strategies and integrated Facebook as a supplementary source. When identity matching is difficult, the agent can retrieve complete Facebook profiles, check candidates' original names, and use location, education, and employer information to help discover and rank LinkedIn candidates. Facebook information is used only for identity discovery, while LinkedIn MCP remains the primary source of truth for final verification. I also implemented concurrent processing, timeouts, and automatic retries for uncertain results. The system uses a 27-minute overall execution budget with time reserved for graceful termination and saving results, without forcibly terminating the process. The agent assigns a score of 0.95 to verified genuine CVs, 0.08 to CVs with confirmed discrepancies, and 0.5 when verification is incomplete. In previous runs on the public_test dataset, my agent correctly classified all 7 CVs, achieving 100% accuracy (7/7) for every runs. 
### Task 2
For Task 2, I used a PDF visual-text inconsistency attack. I modified the PDF so that the visible CV shows an MSc in Finance instead of the actual BSc in Finance, and also includes an additional Python skill. However, the underlying PDF text layer still contains the original information. This creates a difference between what a human sees and what a text extraction tool reads. The attack targets CV verification agents that rely only on extracted PDF text without checking the rendered page. Such agents may read the correct BSc degree, compare it with the LinkedIn MCP record, and incorrectly classify the visually misleading CV as genuine. I expect this technique to work against text-only verification pipelines, although its effectiveness may be reduced if an agent uses OCR or visual inspection. All modifications were made within the PDF, without changing the verification code or MCP data.
