import llm
from config import MAX_STEPS, ENABLE_METRICS
from metrics import MetricsTracker


system_prompt = "You are a professional assistant that can help with various tasks. Your work scope is strictly limited to the workspace. Before modifying any files and codes, you must ask the user for permission adn understan context first. Prefer minimal changes, if a local replacement is enough, do not modify the whole file. After completing the modification, try to run comands to verify the result whenver possible. In your final response to the user: what you changed, which tools you used, and how the verification went."

class BaseAgent:
    def __init__(self,system_prompt:str, tools:list= None, max_steps:int = MAX_STEPS, metrics_tracker:MetricsTracker = None, agent_role:str = "agent"):
        self.system_prompt = system_prompt
        self.tools = tools if tools is not None else []
        self.max_steps = max_steps
        self.metrics_tracker = metrics_tracker
        self.agent_role = agent_role
        self.messages = []


    

    def run(self, input_text:str)->dict:
        self.messages.append({"role":"user", "content":input_text})

        total_input_tokens= 0
        total_output_tokens = 0
        total_latency = 0
        steps_used = 0

        for step in range(self.max_steps):
            response = llm.chat(
                messages = self.messages,
                system_prompt = self.system_prompt,
                tools = self.tools
            )
            total_input_tokens += response.get("input_tokens", 0)
            total_output_tokens += response.get("output_tokens", 0)
            total_latency += response.get("latency", 0)
            steps_used = step + 1


            tool_calls = response.get("tool_calls", [])
            text = response.get("text", "")

            if not tool_calls:
                if text:
                    self.messages.append({"role":"assistant", "content":text})

                self._record_metrics(steps_used, total_input_tokens,total_output_tokens, total_latency)
                return {"text":text, "completed":True}

            #has tool calls ->execute and return results

            self.messages.append(llm.build_assistant_message(response))

            calls_and_results = []
            for call in tool_calls:
                result = self._execute_tool(call['name'], call['args'])
                calls_and_results.append((call, result))

            #append tool results
            self.messages.extend(llm.build_tool_result_message(calls_and_results)) 

            print(f"[DEBUG] messages count: {len(self.messages)}")
            print(f"[DEBUG] last message role: {self.messages[-1]['role']}")

        output_text = f"agent reached max step {self.max_steps} without completing."
        self._record_metrics(steps_used, total_input_tokens, total_output_tokens, total_latency)

        return{"text":output_text, "completed":False}




    def _record_metrics(self,steps:int, input_tokens:int, output_tokens:int, latency:float):
        if ENABLE_METRICS and self.metrics_tracker:
            self.metrics_tracker.record(
                step=steps,
                agent_role=self.agent_role,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency=latency
            )

    def _execute_tool(self, tool_name:str, tool_args:dict)->str:
        """Execute tool, subclasses can override for extra processing"""
        from tool import execute_tool
        print(f"[Tool]{tool_name}({tool_args})")
        result= execute_tool(tool_name, tool_args)
        print(f"[Result]{result[:200]}")
        return result


    def reset_message(self):
        """Clear message history, used for retry senarios"""
        self.messages = []
    

            

        

