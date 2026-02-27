# Code Challenge: The "ME Engineering Assistant" Agent

## Objective

Design, build, and package a multi-tool AI agent that can assist our engineers by answering questions about different Electronic Control Unit (ECU) specifications. The agent must be able to intelligently select the appropriate documentation to consult based on the user's query.

This challenge will assess your ability to design and implement robust RAG and agentic workflows, and to package a solution for production using MLOps best practices.

## Scenario

An engineer at "ME" needs to quickly cross-reference information from two different product lines: the **ECU-700 Series** and the **ECU-800 Series**.

The goal is to build an agent that, when asked a question like *"What is the maximum operating temperature for the ECU-800b?"* or *"Compare the CAN bus speed of the ECU-750 and the ECU-850"*, can autonomously retrieve and synthesize the necessary information to provide a coherent answer.

## Technical Stack & Constraints

* **Language:** Python
* **Core Logic:** LangChain & LangGraph
* **Model Tracking:** MLflow
* **Vector Storage:** An in-memory vector store (e.g., FAISS, ChromaDB)
* **Models & Environment:** You are free to use local or open-source LLM and Embedding models (e.g., from Hugging Face).
* **Packaging & Deployment:** The final solution must be packaged as a containerized application (e.g., using Docker) that serves the MLflow model.
* **Strategy:** Adhere to "everything-as-code" and "python-package-first" principles. The final solution should not be a monolithic notebook, but a proper installable python package

---

## Challenge Structure & Timeline

**Total Time:** We recommend spending **5-6 hours** on the core task. The optional challenges can take an additional **3-5 hours**. You have **10 days** to complete the challenge.

This challenge is divided into a core engineering task and a set of optional advanced challenges that allow you to demonstrate further expertise.

### **Core Engineering Task (Mandatory)**
*Essential skills - must be demonstrated for passing grade*

**Requirements:**
- Functional multi-source RAG system with ECU-700/800 document retrieval
- Working LangGraph agent with intelligent query routing
- Basic MLflow model logging with predict() method
- Clear architectural documentation and design

**Success Criteria:**
- Agent correctly answers 8/10 predefined test queries
- Response time <20 seconds per query
- Code passes basic quality checks (pylint score > 85%)

**Recommended Time:** 5-6 hours

### **Optional Advanced Challenges**
*Demonstrate your production-readiness and technical leadership by choosing one or more of the following challenges.*

#### **1. Production & MLOps Excellence**
- The agent is logged as a custom MLflow model.
- The application is containerized (e.g., using Docker) to serve the MLflow model.
- The agent is exposed via a REST API (e.g., using `mlflow models serve`).
- Comprehensive testing and validation strategy (documented).

#### **2. Innovation & Leadership**
- Implemented evaluation framework with MLflow evaluation
- Human-in-the-loop mechanisms for low-confidence scenarios
- Scalability strategy with detailed implementation plan
- Advanced agent behaviors (multi-step reasoning, tool use)

---

## Core Deliverables

Your final submission must be a well-structured project that includes:

### **1. Containerized Application**
The complete LangGraph agent, packaged as a containerized application (e.g., using Docker) that exposes a REST API for predictions.

### **2. Source Code**
The complete source code for the agent, including any scripts for building and running the application.

### **3. Comprehensive Documentation**
A project `README.md` that serves as the single source of truth, including:

- **Architectural Design:** High-level design and key decisions (chunking strategy, agent graph structure, etc.)
- **Setup & Deployment:** Clear instructions for building and running the containerized application.
- **Testing & Validation Strategy:** Conceptual framework for production validation, including:
  - Proposed evaluation metrics and automated testing approaches
  - Domain expertise validation methods (e.g., using golden datasets or domain-specific benchmarks)
  - Strategies for continuous validation and monitoring of agent performance in production
- **Limitations & Future Work:** Discussion of approach limitations and potential improvements

---

## What We Provide

### **Infrastructure & Access**
- Small dataset of sample documents (`ECU-700-manual.md`, `ECU-800a.md`, `ECU-800b.md`)

### **Technical Fallback Options**
- **Vector Store Issues:** Since documents are relatively small, you can implement a fallback strategy that passes document content directly as context to the LLM
- **Alternative Frameworks:** If LangGraph proves challenging, alternative agent frameworks are allowed with proper justification

---

## Evaluation Criteria

### **Technical Excellence**
1. **Functionality & Robustness:** Does the agent correctly answer different types of queries? How does it handle edge cases?
2. **Code Quality:** Is the solution well-designed, modular, and maintainable? We value clean code and sound software engineering principles.

### **Production Readiness (Optional)**
3. **MLOps Integration:** How effectively have you used MLflow for model packaging and serving?
4. **Containerization and API Design:** How effectively is the application containerized and exposed via a REST API?

### **Strategic Thinking**
5. **Architectural Decision-Making:** Your documentation should demonstrate informed design trade-offs and clear articulation of your approach.
6. **Testing Strategy:** How comprehensive and practical is your proposed validation framework?

### **Communication**
7. **Final Presentation:** 60-minute code review and discussion where you present your solution, rationale, and lessons learned.

---

## Getting Started

1. **Environment Setup:** Set up your local development environment.
2. **Architecture Planning:** Design your approach for the agent.
3. **Iterative Development:** Build your solution incrementally, focusing on the core requirements first.
4. **Documentation:** Maintain clear documentation throughout the development process.

## Support & Questions

- **Technical Issues:** Contact us via email for environment/tooling questions.
- **Clarifications:** Don't hesitate to ask about requirements or expectations.

---

**Timeline:** You have **10 days** to complete this challenge. We recommend distributing the hours across the timeline to allow for iterative development and refinement.

**Submission:** Provide repository access to your solution. If you complete the optional challenges, please ensure the containerized application is runnable.

Good luck! We're excited to see your approach to building production-ready AI engineering solutions.