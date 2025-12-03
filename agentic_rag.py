import os
from datetime import datetime
import random

import requests
import os
from dotenv import load_dotenv
from crewai import Agent, Crew, Task, Process, LLM
from crewai.tools import BaseTool

# 加载配置
load_dotenv()
# from opensearch_client import get_opensearch_client
from patent_crew import run_patent_analysis


# 注释掉OpenSearch相关的搜索函数，替换为本地模拟函数
# from patent_search_tools import hybrid_search, iterative_search, semantic_search, keyword_search


# 模拟搜索结果（替代OpenSearch的搜索功能）
def mock_search_results(query):
    """模拟专利搜索结果，避免OpenSearch依赖"""
    mock_patents = [
        {
            "_source": {
                "title": f"Lithium Battery {query} Technology",
                "abstract": f"Novel {query} technology for lithium battery energy storage, improving cycle life by 30% and energy density by 25%. This patent discloses a new electrode material and manufacturing process suitable for electric vehicle applications.",
                "publication_date": "2024-01-15",
                "patent_id": f"CN2024{random.randint(100000, 999999)}"
            },
            "_score": round(random.uniform(80, 99), 2)
        },
        {
            "_source": {
                "title": f"High-Efficiency {query} for Lithium-Ion Batteries",
                "abstract": f"Optimized {query} structure for lithium-ion batteries, reducing internal resistance and improving charge/discharge efficiency. Applicable to consumer electronics and energy storage systems.",
                "publication_date": "2023-10-22",
                "patent_id": f"CN2023{random.randint(100000, 999999)}"
            },
            "_score": round(random.uniform(75, 89), 2)
        },
        {
            "_source": {
                "title": f"Environmental Protection {query} in Lithium Battery Production",
                "abstract": f"Green {query} technology for lithium battery production, reducing carbon emissions by 40% and waste generation by 35%. Complies with international environmental standards and reduces production costs.",
                "publication_date": "2024-03-08",
                "patent_id": f"CN2024{random.randint(100000, 999999)}"
            },
            "_score": round(random.uniform(70, 85), 2)
        }
    ]
    return mock_patents


def display_menu():
    """Display the main menu options"""
    print("\n" + "=" * 60)
    print("  PATENT INNOVATION PREDICTOR - LITHIUM BATTERY TECHNOLOGY  ")
    print("=" * 60)
    print("1. Run complete patent trend analysis and forecasting")
    print("2. Search for specific patents (mock data)")
    print("3. Iterative patent exploration (mock data)")
    print("4. View system status (skip OpenSearch check)")
    print("5. Exit")
    print("-" * 60)
    return input("Select an option (1-5): ")


def run_complete_analysis():
    """Run the complete patent trend analysis using CrewAI agents"""
    print("\nRunning comprehensive patent analysis...")
    print("This may take several minutes depending on the data volume.")

    research_area = input("Enter research area (default: Lithium Battery): ")
    if not research_area:
        research_area = "Lithium Battery"

    # Ask for the Ollama model to use
    model_name = "deepseek-chat"
    
    print(f"\nAnalyzing patents for: {research_area}")
    print(f"Using Ollama model: {model_name}")
    print("Agents are now processing the data...\n")

    try:
        result = run_patent_analysis(research_area, model_name)

        # Ensure result is a string before writing to file
        if not isinstance(result, str):
            result = str(result)

        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"patent_analysis_{timestamp}.txt"
        with open(filename, "w", encoding="utf-8") as f:  # 添加编码避免中文乱码
            f.write(result)

        print(f"\nAnalysis completed and saved to {filename}")

        # Display summary
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("-" * 60)
        print(result[:500] + "...\n")  # Display first 500 chars

    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Tips: Ensure your .env file has the correct OPENAI_API_KEY for DeepSeek and you have an internet connection.")


def search_patents():
    """Search for specific patents (mock data, no OpenSearch)"""
    print("\nPATENT SEARCH (MOCK DATA)")
    print("-" * 60)

    query = input("Enter search query: ")
    if not query:
        print("Search query cannot be empty.")
        return

    search_type = input("Select search type (1: Keyword, 2: Semantic, 3: Hybrid) [3]: ")
    if not search_type:
        search_type = "3"

    try:
        # 使用模拟数据替代真实搜索
        results = mock_search_results(query)

        # 模拟不同搜索类型的提示
        search_type_name = {
            "1": "Keyword",
            "2": "Semantic",
            "3": "Hybrid"
        }.get(search_type, "Hybrid")

        print(f"\n[{search_type_name} Search] Found {len(results)} results for '{query}':")
        print("-" * 60)
        for i, hit in enumerate(results):
            source = hit["_source"]
            print(f"{i + 1}. {source['title']}")
            print(f"   Score: {hit['_score']}")
            print(f"   Date: {source.get('publication_date', 'N/A')}")
            print(f"   Patent ID: {source.get('patent_id', 'N/A')}")
            print(f"   Abstract: {source['abstract'][:150]}...")
            print("-" * 60)

    except Exception as e:
        print(f"Search error: {e}")


def iterative_exploration():
    """Perform iterative exploration of patents (mock data)"""
    print("\nITERATIVE PATENT EXPLORATION (MOCK DATA)")
    print("-" * 60)

    query = input("Enter initial exploration query: ")
    if not query:
        print("Query cannot be empty.")
        return

    steps = input("Number of exploration steps (default: 3): ")
    try:
        steps = int(steps) if steps else 3
    except:
        steps = 3

    print(f"\nExploring patents related to '{query}' with {steps} refinement steps...")

    try:
        # 使用模拟数据替代真实迭代搜索
        results = mock_search_results(query)

        # 模拟迭代优化后的结果（重复数据但修改分数）
        for hit in results:
            hit["_score"] = round(hit["_score"] * (1 + steps * 0.1), 2)

        print(f"\nFound {len(results)} results through iterative exploration:")
        print("-" * 60)
        for i, hit in enumerate(results):
            source = hit["_source"]
            print(f"{i + 1}. {source['title']}")
            print(f"   Date: {source.get('publication_date', 'N/A')}")
            print(f"   Patent ID: {source.get('patent_id', 'N/A')}")
            print(f"   Abstract: {source['abstract'][:150]}...")
            print("-" * 60)

    except Exception as e:
        print(f"Exploration error: {e}")


def check_system_status():
    """Check the status of system components"""
    print("\nSYSTEM STATUS")
    print("-" * 60)
    # 检查环境变量
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("✅ DeepSeek API Key: Loaded from .env file.")
    else:
        print("❌ DeepSeek API Key: Not found. Please check your .env file.")
    # 检查到 DeepSeek API 的网络连接
    try:
        response = requests.get("https://api.deepseek.com", timeout=5)
        if response.status_code < 400: # 任何成功的状态码
             print("✅ DeepSeek API Connection: OK (Network reachable)")
        else:
            print(f"❌ DeepSeek API Connection: Failed (Status code: {response.status_code})")
    except requests.exceptions.RequestException as e:
        print(f"❌ DeepSeek API Connection: Failed - {e}")
        print("   Tips: Check your internet connection.")
    print("\nSystem is ready for operation.")

    # Check embedding model (简化检查)
    try:
        from embedding import get_embedding
        sample = get_embedding("test")
        print(f"✅ Embedding model: OK (dimension: {len(sample)})")
    except Exception as e:
        print(f"ℹ️ Embedding model: Check skipped - {e}")

    print("\nSystem is ready for operation (using mock data for search functions).")


def main():
    """Main application entry point"""
    # Load environment variables (still useful for other potential secrets)
    load_dotenv()

    print("Welcome to Patent Innovation Predictor!")
    print("Note: Using mock data for search functions (no OpenSearch required)")

    while True:
        choice = display_menu()

        if choice == "1":
            run_complete_analysis()
        elif choice == "2":
            search_patents()
        elif choice == "3":
            iterative_exploration()
        elif choice == "4":
            check_system_status()
        elif choice == "5":
            print("\nExiting Patent Innovation Predictor. Goodbye!")
            break
        else:
            print("\nInvalid option. Please select a number between 1 and 5.")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()