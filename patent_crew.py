import os
from datetime import datetime
from typing import List, Optional

# 适配CrewAI 0.28.8 + Ollama
from crewai import Agent, Crew, Task, Process, LLM
from crewai.tools import BaseTool
from dotenv import load_dotenv
# 引入ChatOllama显式配置本地模型
# from langchain_ollama import ChatOllama


# 模拟专利数据生成（避免外部依赖）
def generate_mock_patent_data(research_area: str) -> List[dict]:
    """生成模拟专利数据，用于分析"""
    mock_data = [
        {
            "title": f"{research_area} Electrode Material Optimization",
            "abstract": f"Novel electrode material for {research_area} applications, improving cycle life by 35% and energy density by 28%. The material uses nanocomposite technology and is suitable for high-performance EV batteries.",
            "publication_date": "2024-01-15",
            "inventor": "Zhang San",
            "assignee": "Battery Tech Co., Ltd.",
            "tech_field": "Electrode Materials"
        },
        {
            "title": f"{research_area} Thermal Management System",
            "abstract": f"Intelligent thermal management system for {research_area} packs, reducing operating temperature by 15°C and improving safety by 40%. Integrated with AI-based temperature prediction algorithms.",
            "publication_date": "2024-02-20",
            "inventor": "Li Si",
            "assignee": "New Energy Auto Group",
            "tech_field": "Thermal Management"
        },
        {
            "title": f"{research_area} Recycling Technology",
            "abstract": f"Eco-friendly {research_area} recycling process, recovering 98% of lithium and cobalt materials. Reduces environmental impact and raw material costs by 30%.",
            "publication_date": "2024-03-10",
            "inventor": "Wang Wu",
            "assignee": "Recycling Tech Inc.",
            "tech_field": "Recycling & Sustainability"
        },
        {
            "title": f"{research_area} Solid-State Battery Design",
            "abstract": f"Next-generation solid-state {research_area} design, eliminating liquid electrolyte and improving energy density by 50%. Achieves 1000+ charge/discharge cycles with no safety risks.",
            "publication_date": "2024-04-05",
            "inventor": "Zhao Liu",
            "assignee": "Advanced Battery Lab",
            "tech_field": "Solid-State Batteries"
        }
    ]
    return mock_data


# 修复Tool类定义（适配CrewAI 0.28.8 + Pydantic V1）
class PatentSearchTool(BaseTool):
    name: str = "Patent Search Tool"
    description: str = "Searches for patent data related to the research area"

    def _run(self, research_area: str) -> List[dict]:
        """执行专利数据搜索（模拟）"""
        return generate_mock_patent_data(research_area)

    # 新增required_fields，解决Pydantic验证缺失func的问题
    @property
    def required_fields(self):
        return ["research_area"]


def run_patent_analysis(research_area: str, model_name: str = "deepseek-chat") -> str:
    """
    运行专利分析（基于CrewAI agents，使用Ollama模型）
    """
    # 加载环境变量
    load_dotenv()

    # 核心修复：显式配置Ollama LLM（替代字符串配置）
    llm = LLM(
        model=model_name,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.deepseek.com"
    )

    # 1. 创建Agent：专利研究员
    patent_researcher = Agent(
        role="Senior Patent Researcher",
        goal=f"Collect and analyze the latest {research_area} patent data",
        backstory="Expert in patent analysis with 10+ years of experience in battery technology. Specializes in identifying technical trends and innovation opportunities.",
        verbose=True,
        allow_delegation=False,
        tools=[PatentSearchTool()],
        llm=llm  # 传入配置好的LLM对象
    )

    # 2. 创建Agent：技术趋势分析师
    trend_analyst = Agent(
        role="Technology Trend Analyst",
        goal=f"Identify key trends and forecasting in {research_area} patents",
        backstory="Data scientist specializing in emerging technology trends. Expert in predicting market adoption and technical breakthroughs.",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # 3. 创建Agent：报告撰写人
    report_writer = Agent(
        role="Technical Report Writer",
        goal=f"Compile a comprehensive analysis report for {research_area} patents",
        backstory="Professional technical writer with experience in battery technology reports. Skilled at translating complex data into clear insights.",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # 4. 创建任务1：收集专利数据
    task1 = Task(
        description=f"Search for the latest {research_area} patents (2023-2024) and extract key information: title, abstract, technical field, publication date, innovation points.",
        agent=patent_researcher,
        expected_output="List of 4+ patents with detailed technical information and innovation highlights.",
        inputs={"research_area": research_area}
    )

    # 5. 创建任务2：分析技术趋势
    task2 = Task(
        description=f"Analyze the patent data to identify key technical trends in {research_area} field: emerging technologies, top innovators, performance improvements, market opportunities.",
        agent=trend_analyst,
        expected_output="Detailed trend analysis with data-backed insights and 3-5 key findings."
    )

    # 6. 创建任务3：撰写分析报告
    task3 = Task(
        description=f"Compile a comprehensive patent analysis report for {research_area} technology, including executive summary, technical trends, innovation opportunities, and future forecasting (next 3-5 years).",
        agent=report_writer,
        expected_output="Full analysis report in natural language, professional and easy to understand, 800-1000 words."
    )

    # 7. 创建Crew并运行
    crew = Crew(
        agents=[patent_researcher, trend_analyst, report_writer],
        tasks=[task1, task2, task3],
        process=Process.sequential,  # 顺序执行任务
        verbose=True  # 显示详细执行过程
    )

    # 运行分析（传入全局输入参数）
    print(f"\nStarting {research_area} patent analysis with {model_name} model...")
    result = crew.kickoff(inputs={"research_area": research_area})

    # 格式化结果（确保result是字符串）
    if not isinstance(result, str):
        result = str(result)

    # 格式化最终报告
    final_report = f"""
# {research_area} Patent Trend Analysis Report
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Model used: {model_name}

{result}

---
Note: This analysis is based on mock patent data (no real OpenSearch integration).
For production use, integrate with real patent database/OpenSearch.
"""

    return final_report


# 测试函数（可选）
if __name__ == "__main__":
    test_result = run_patent_analysis("Lithium Battery", "deepseek-chat")
    print(test_result)