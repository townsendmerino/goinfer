import json, sys, urllib.request
URL=sys.argv[1]; MODEL=sys.argv[2]
def post(body):
    r=urllib.request.urlopen(urllib.request.Request(URL+"/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=300)
    return json.load(r)
tools=[{"type":"function","function":{"name":"get_weather","description":"Get the current weather in a given city.","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]
msgs=[{"role":"user","content":"What is the weather in Paris? Use the get_weather tool."}]
r1=post({"model":MODEL,"temperature":0,"max_tokens":96,"messages":msgs,"tools":tools})
m1=r1["choices"][0]["message"]; print("TURN1 content:",repr(m1.get("content")),"tool_calls:",json.dumps(m1.get("tool_calls")),"finish:",r1["choices"][0]["finish_reason"])
tc=m1.get("tool_calls") or []
if tc:
    msgs+= [{"role":"assistant","content":None,"tool_calls":[{"id":tc[0]["id"],"type":"function","function":tc[0]["function"]}]},{"role":"tool","tool_call_id":tc[0]["id"],"content":'{"temp_c":14,"sky":"rain"}'}]
    r2=post({"model":MODEL,"temperature":0,"max_tokens":96,"messages":msgs,"tools":tools})
    m2=r2["choices"][0]["message"]; print("TURN2 content:",repr(m2.get("content")),"tool_calls:",json.dumps(m2.get("tool_calls")),"finish:",r2["choices"][0]["finish_reason"])
