import braintrust
from typing import *
from lm_act_eval.ontology import EvaluationLogs

class BraintrustLogger:
    def __init__(self, project_name: str, braintrust_api_key: str):
        self.experiment = braintrust.init(project=project_name, api_key=braintrust_api_key)
    
    @property
    def summary(self):
        # This property returns a summary of the experiment session
        return self.experiment.summarize()

    def log_evaluation(self, evaluation_input: EvaluationLogs, metadata: Optional[Dict] = None):
        """
        Logs evaluation results to BrainTrust using the defined ontology.
        
        :param evaluation_input: An instance of EvaluationLogs containing the evaluation details.
        :param metadata: Optional dictionary containing additional metadata for the log.
        """
        log_data = {
            'inputs': {"query": evaluation_input.input},
            'output': evaluation_input.output,
            'expected': evaluation_input.expected,
            'scores': evaluation_input.scores,
        }
        
        if metadata is not None:
            log_data['metadata'] = metadata

        self.experiment.log(**log_data)

# Example usage
if __name__ == "__main__":
    logger = BraintrustLogger(project_name="Autoevals", braintrust_api_key="YOUR_BRAINTRUST_API_KEY")
    
    # Create an instance of EvaluationLogs
    evaluation_input = EvaluationLogs(
        input="Which country has the highest population?",
        output="People's Republic of China",
        expected="China",
        scores={'factuality': 1.0}  # Example of additional evaluation metric
    )
    
    # Optional metadata
    metadata = {'factuality': {'rationale': "Based on population data"}}

    # Log the evaluation using the defined ontology
    logger.log_evaluation(evaluation_input, metadata=metadata)
    
    # Print the summary
    print(logger.summary)
