# Contributing to Research Agent

👋 **Welcome!** We are so happy you are here.

First things first: If you have never contributed to open source before, you are in the right place. The Research Agent is built on the idea that we are all learning. We don't expect perfection; we value curiosity and the willingness to try. Whether you are fixing a typo, updating documentation, or writing complex code, your contribution matters.

---

## 🌟 Our Core Principles

Before you start, it helps to know what we care about. We use these principles to make decisions about code and features:

- **Useful Tools:** We build things that are practical and easy to adopt.
- **Public Benefit:** We prioritize safety over speed. Does this help humans?
- **Open by Default:** We work in the open.
- **Privacy First:** We never collect or store personal user data. We do not track users.
- **Humility:** We own our mistakes and pivot when we learn new things.

---

## 🔍 How to Find Something to Work On

You don't need a grand plan to contribute. Here are the best ways to start:

- **Join our slack channel and/or jira board.** We have a backlog of issues that you can pick up. Or simply create a new issue yourself.
  - **Slack channel:** https://os4g.slack.com
  - **Jira board:** https://nurtekinsavasai.atlassian.net/jira/software/projects/BTS/boards/1

- **Documentation:** Found a confusing sentence in the README? A typo? Fixing documentation is one of the most valuable contributions you can make.

- **Bug Reports:** If something isn't working, let us know! Open an Issue describing what happened. Either in github, slack or JIRA

- **Join by emailing** nurtekinsavas@gmail.com. We guarantee a response!

---

## 🛠️ How to Submit Your First Contribution (The "PR" Process)

In open source, we use a process called a **Pull Request (PR)**. Think of a PR as "packaging up your changes and asking us to pull them into the main project."

Here is the step-by-step guide for beginners:

### Step 1: "Fork" the Repository

You can't edit this project directly (yet!). You need your own copy.

1. Look for the button that says **Fork** in the top right corner of this page.
2. Click it. This creates a copy of Research Agent in your account.

### Step 2: "Clone" it to your computer

Now you need to get that copy onto your machine.

1. Click the green **Code** button on your forked copy.
2. Copy the URL.
3. Open your terminal and type:

```bash
git clone [PASTE THE URL HERE]
```

### Step 3: Create a "Branch"

Never work on the main branch directly. Think of a branch as a sandbox where you can build without breaking anything.

1. Go into the folder: `cd arxiv-ai-agent-v2`
2. Create a branch with a descriptive name:

```bash
git checkout -b fix-typo-in-readme
```

### Step 4: Make your changes

Open the files in your favorite text editor (VS Code, Sublime, etc.) and make your edits.

### Step 5: Save (Commit) your changes

Tell git which files you changed:

```bash
git add .
```

Save them with a message explaining what you did:

```bash
git commit -m "Fixed a typo in the introduction"
```

### Step 6: Push your changes

Send your changes from your computer up to your copy on GitHub:

```bash
git push origin fix-typo-in-readme
```

### Step 7: Open the Pull Request

1. Go to the original Research Agent page.
2. You will likely see a yellow banner saying "Compare & Pull Request." Click it!
3. Write a title and description. Be honest—if you aren't sure if it works perfectly, just say so! We are here to help.
4. Click **Create Pull Request**.

🎉 **You did it!** You just made an open source contribution.

---

## 🤝 Code of Conduct (How we treat each other)

We are committed to providing a friendly, safe, and welcoming environment for everyone, regardless of level of experience, gender identity and expression, sexual orientation, disability, personal appearance, body size, race, ethnicity, age, religion, or nationality.

### Our Standards

- **Be Kind:** We prioritize kindness. Critique ideas, not people.
- **Be Humble:** If someone suggests a better way, listen. If you make a mistake, own it. We all make mistakes.
- **No Harassment:** We have zero tolerance for harassment or abuse of any kind.
- **Privacy is Sacred:** Do not submit code that adds tracking, telemetry, or data collection. It will be rejected.

### Enforcement

If you see behavior that violates this code of conduct, please report it to the maintainers. We promise to listen and take action to keep this community safe.

---

## ❓ Need Help?

If you get stuck at any point—git is confusing, the code doesn't run, or you're just scared to push the button—please open an Issue and label it "Question."

We remember what it was like to be new. We will help you.

