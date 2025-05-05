class Context:
    def __init__(self, query):
        self.original_query = query
        self.subquestions = []
        self.papers = []
        self.summaries = []
        self.hypotheses = []
        self.status = "STARTED"

    def add_subquestion(self, question):
        self.subquestions.append({"question": question, "status": "pending"})

    def complete_subquestion(self, index, summary):
        self.subquestions[index]["status"] = "completed"
        self.summaries.append(summary)

    def add_paper(self, paper_metadata):
        self.papers.append(paper_metadata)

    def add_hypothesis(self, hypothesis):
        self.hypotheses.append(hypothesis)

    def __str__(self):
        return f"Query: {self.original_query}\nSubquestions: {self.subquestions}\nSummaries: {self.summaries}\nHypotheses: {self.hypotheses}"