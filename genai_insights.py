import os
import json
import asyncio
from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from typing import List, Dict, Union

client= OpenAIChatCompletionClient(
    model="gemini-1.5-flash",
    api_key="AIzaSyC8Q7GccF-P1b_AIAcwuZNHsA0Psou5mXs"
)


async def get_sales_insights_and_recommendations(forecast_data: List[Dict], mode: str = "single", group_name: str = None, group_value: str = None, target: float = None) -> Dict[str, Union[str, List[str]]]:
    """
    Given forecasted data from Prophet, return GenAI sales insights and recommendations using Gemini 1.5 Flash.
    
    Args:
        forecast_data (List[Dict]): List of forecast records with keys: ds, yhat, yhat_lower, yhat_upper
        mode (str): 'single' for normal, 'consolidated' for group analysis
        group_name (str): Name of the group (e.g., 'Sales Head')
        group_value (str): Value of the group (e.g., 'John Doe')

    Returns:
        Dict with:
            'genai_insights': Sales insights in business language (str)
            'recommendations': List of actionable recommendations (List[str])
    """
    # Convert all Timestamps to str for JSON serialization
    for row in forecast_data:
        if 'ds' in row and hasattr(row['ds'], 'isoformat'):
            row['ds'] = row['ds'].isoformat()

    if mode == "consolidated":
        prompt = (
            f"You are a senior sales analytics expert. Analyze the following forecasted sales data for the group: {group_name} = {group_value}. "
            "Use sales language and provide:\n"
            "1. 3-5 numbered, concise, and actionable sales insights in bullet points, using numbers and percentages where possible (e.g., 'Sales grew by 12% in Q2').\n"
            "2. Each insight should reference the group (Sales Head, Regional Manager, Product, or Location) and be relevant to business decisions.\n"
            "3. Output strictly as JSON in this format:\n"
            '{\n  "genai_insights": ["insight 1", "insight 2", ...],\n  "recommendations": ["rec 1", "rec 2", ...]\n}\n\n'
            f"Forecasted Data (first 10 rows): {json.dumps(forecast_data[:10], indent=2)}"
        )
    elif mode == "summary":
        prompt = (
            f"You are a senior sales analytics expert. Analyze the following forecasted sales data for the group: {group_name} = {group_value}. "
            f"The sales target for this group is {target}.\n"
            "Use sales language and provide:\n"
            "1. A one-line insight about the trend and target (mention if the forecast meets or misses the target, use numbers/percentages, do not mention time or (start: ...)).\n"
            "2. A one-line forecast summary (no time, just the value and target comparison).\n"
            "3. A one-line actionable recommendation to improve or sustain performance, using numbers/percentages.\n"
            "Output strictly as JSON in this format:\n"
            '{\n  "insight": "...",\n  "forecast": "...",\n  "recommendation": "..."\n}\n\n'
            f"Forecasted Data (first 10 rows): {json.dumps(forecast_data[:10], indent=2)}"
        )
    else:
        prompt = (
            "You are a sales analytics expert. Given the following forecasted sales data (columns: ds, yhat, yhat_lower, yhat_upper), "
            "analyze the trends, growth, and risks. Provide:\n"
            "1. A concise sales insight summary in business language (max 100 words).\n"
            "2. 3 actionable recommendations for the sales team.\n\n"
            "Output strictly as JSON in this format:\n"
            '{\n  "genai_insights": "insight text...",\n  "recommendations": ["rec 1", "rec 2", "rec 3"]\n}\n\n'
            f"Forecasted Data (first 10 rows): {json.dumps(forecast_data[:10], indent=2)}"
        )

    try:
        response = await client.create([UserMessage(content=prompt, source="user")])
        response_text = _extract_gemini_text(response)
        result = json.loads(response_text)
        if mode == "consolidated":
            if isinstance(result, dict) and 'genai_insights' in result and 'recommendations' in result:
                return result
        elif mode == "summary":
            if isinstance(result, dict) and 'insight' in result and 'forecast' in result and 'recommendation' in result:
                return result
        else:
            if isinstance(result, dict) and 'genai_insights' in result and 'recommendations' in result:
                return result
    except Exception as e:
        print(f"[Warning] Failed to parse Gemini response as JSON: {e}")

    return {
        "genai_insights": str(response) if 'response' in locals() else "No response received",
        "recommendations": []
    }

async def get_genai_single_insights(forecast_data):
    from autogen_core.models import UserMessage
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    client = OpenAIChatCompletionClient(model="gemini-1.5-flash", api_key=os.getenv("GEMINI_API_KEY", "AIzaSyBvNs3gu-97GrpcFIGyED4QbJZIJrriZn8"))
    # Convert timestamps
    for row in forecast_data:
        if 'ds' in row and hasattr(row['ds'], 'isoformat'):
            row['ds'] = row['ds'].isoformat()
    prompt = (
        "You are a sales analytics expert. Given the following forecasted sales data (columns: ds, yhat, yhat_lower, yhat_upper), "
        "analyze the trends, growth, and risks. Provide:\n"
        "1. A concise sales insight summary in business language (max 100 words).\n"
        "2. 3 actionable recommendations for the sales team.\n\n"
        "Output strictly as JSON in this format:\n"
        '{\n  "genai_insights": "insight text...",\n  "recommendations": ["rec 1", "rec 2", "rec 3"]\n}\n\n'
        f"Forecasted Data (first 10 rows): {json.dumps(forecast_data[:10], indent=2)}"
    )
    response = await client.create([UserMessage(content=prompt, source="user")])
    try:
        response_text = _extract_gemini_text(response)
        result = json.loads(response_text)
        if isinstance(result, dict) and 'genai_insights' in result and 'recommendations' in result:
            return result
    except Exception as e:
        print(f"[Warning] Failed to parse Gemini response as JSON: {e}")
    return {"genai_insights": str(response) if 'response' in locals() else "No response received", "recommendations": []}

async def get_genai_consolidated_insights(df, group_cols, periods):
    from utils import filter_sales_data, forecast_sales
    from autogen_core.models import UserMessage
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    client = OpenAIChatCompletionClient(model="gemini-1.5-flash", api_key=os.getenv("GEMINI_API_KEY", "AIzaSyBvNs3gu-97GrpcFIGyED4QbJZIJrriZn8"))
    results = {}
    for col in group_cols:
        results[col] = {}
        for val in df[col].dropna().unique():
            filtered = filter_sales_data(df, {col: [val]})
            if filtered.empty:
                continue
            forecast_df = forecast_sales(filtered, periods)
            forecast_data = forecast_df.to_dict(orient="records")
            for row in forecast_data:
                if 'ds' in row and hasattr(row['ds'], 'isoformat'):
                    row['ds'] = row['ds'].isoformat()
            prompt = (
                f"You are a senior sales analytics expert. Analyze the following forecasted sales data for the group: {col} = {val}. "
                "Use sales language and provide:\n"
                "1. 3-5 numbered, concise, and actionable sales insights in bullet points, using numbers and percentages where possible.\n"
                "2. Each insight should reference the group and be relevant to business decisions.\n"
                "3. Output strictly as JSON in this format:\n"
                '{\n  "genai_insights": ["insight 1", "insight 2", ...],\n  "recommendations": ["rec 1", "rec 2", ...]\n}\n\n'
                f"Forecasted Data (first 10 rows): {json.dumps(forecast_data[:10], indent=2)}"
            )
            response = await client.create([UserMessage(content=prompt, source="user")])
            try:
                response_text = _extract_gemini_text(response)
                result = json.loads(response_text)
                if isinstance(result, dict) and 'genai_insights' in result and 'recommendations' in result:
                    results[col][val] = result
            except Exception as e:
                print(f"[Warning] Failed to parse Gemini response as JSON: {e}")
    return results

async def get_genai_forecast_summary(df, group_cols, periods=30, period_type="days"):
    from utils import filter_sales_data, forecast_sales
    from autogen_core.models import UserMessage
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    client = OpenAIChatCompletionClient(model="gemini-1.5-flash", api_key=os.getenv("GEMINI_API_KEY", "AIzaSyBvNs3gu-97GrpcFIGyED4QbJZIJrriZn8"))
    insights_forecast = []
    recommendations = []
    targets = {"Product": 4000, "Region": 3500, "Sales office": 3000, "Sales Head": 4500}
    for col in group_cols:
        for val in df[col].dropna().unique():
            filtered = filter_sales_data(df, {col: [val]})
            if filtered.empty:
                continue
            forecast_df = forecast_sales(filtered, periods)
            forecast_data = forecast_df.to_dict(orient="records")
            forecast_data = [row for row in forecast_data if row.get('yhat') is not None]
            if not forecast_data:
                continue
            # Use only the last forecast for summary to avoid token limit
            last_row = forecast_data[-1]
            if 'ds' in last_row and hasattr(last_row['ds'], 'isoformat'):
                last_row['ds'] = last_row['ds'].isoformat()
            target = targets.get(col, 4000)
            prompt = (
                f"You are a senior sales analytics expert. Analyze the following forecasted sales data for the group: {col} = {val}. "
                f"The sales target for this group is {target}. The forecast period is {periods} {period_type}.\n"
                "Use sales language and provide:\n"
                "1. A one-line insight about the trend and target (mention if the forecast meets or misses the target, use numbers/percentages, do not mention time or (start: ...)).\n"
                "2. A one-line forecast summary (no time, just the value and target comparison).\n"
                "3. A one-line actionable recommendation to improve or sustain performance, using numbers/percentages.\n"
                "Output strictly as JSON in this format:\n"
                '{\n  "insight": "...",\n  "forecast": "...",\n  "recommendation": "..."\n}\n\n'
                f"Last Forecasted Data: {json.dumps(last_row, indent=2)}"
            )
            try:
                response = await client.create([UserMessage(content=prompt, source="user")])
                response_text = _extract_gemini_text(response)
                result = json.loads(response_text)
                if result and (result.get("insight") or result.get("forecast") or result.get("recommendation")):
                    insights_forecast.append({val: {"Insight": result.get("insight", "No insight generated."), "Forecast": result.get("forecast", "No forecast generated.")}})
                    recommendations.append({val: result.get("recommendation", "No recommendation generated.")})
            except Exception as e:
                print(f"[Warning] Failed to parse Gemini response as JSON: {e}")
    return {"insights_forecast": insights_forecast, "recommendations": recommendations}

def _extract_gemini_text(response):
    """Extract text content from Gemini client response object."""
    # Prefer 'content' attribute for Gemini
    text = getattr(response, 'content', None)
    if isinstance(text, str):
        # Remove markdown code block if present
        if text.strip().startswith('```json'):
            text = text.strip().removeprefix('```json').removesuffix('```').strip()
        elif text.strip().startswith('```'):
            text = text.strip().removeprefix('```').removesuffix('```').strip()
        return text
    # Fallbacks
    if hasattr(response, 'text') and isinstance(response.text, str):
        return response.text
    if hasattr(response, 'choices'):
        choices = getattr(response, 'choices')
        if isinstance(choices, list) and choices:
            msg = choices[0]
            if isinstance(msg, dict):
                if 'message' in msg and isinstance(msg['message'], dict) and 'content' in msg['message']:
                    return msg['message']['content']
                if 'content' in msg:
                    return msg['content']
            elif hasattr(msg, 'message') and hasattr(msg.message, 'content'):
                return msg.message.content
            elif hasattr(msg, 'content'):
                return msg.content
    return str(response)

# Example usage
# asyncio.run(get_sales_insights_and_recommendations(forecast_data))
