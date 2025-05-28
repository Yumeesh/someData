from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import UserMessage
import asyncio

async def test():
    client = OpenAIChatCompletionClient(model='gemini-1.5-flash', api_key='AIzaSyC8Q7GccF-P1b_AIAcwuZNHsA0Psou5mXs')
    resp = await client.create([UserMessage(content='Say hello in JSON format only: {"hello": "world"}', source='user')])
    print('RAW:', resp)
    print('TEXT:', getattr(resp, 'text', None))
    print('CONTENT:', getattr(resp, 'content', None))
    print('CHOICES:', getattr(resp, 'choices', None))
    print('STR:', str(resp))

if __name__ == '__main__':
    asyncio.run(test())
