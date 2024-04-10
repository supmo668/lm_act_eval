# LLM Action Evaluation

## Script use
* Show configuration 
  ```
  python lm_act_eval/_init_conf.py trajectory_eval-dev
  ```
  

## Configuration structure

* *Opentable Example*
```
project: opentable                                                                                                                                                             
eval:                                                                                                                                                                          
  sft:                                                                                                                                                                         
    trajectory:                                                                                                                                                                
      data:                                                                                                                                                                    
        path: lm_act_eval/.cache/five-star-trajectories/csv/data.csv                                                                                                           
        columns:                                                                                                                                                               
          'y': ground_truth                                                                                                                                                    
          y_': GPTV_generations                                                                                                                                                
        extract_fs:                                                                                                                                                            
          QUERY:                                                                                                                                                               
            QUERY: null                                                                                                                                                        
          screenshot:                                                                                                                                                          
            screenshot: null                                                                                                                                                   
          GOAL:                                                                                                                                                                
            chat_completion_messages: parse_completion.parse_content                                                                                                           
          HTML:                                                                                                                                                                
            html: opentable_extract_reservation_details                                                                                                                        
        logging:                                                                                                                                                               
          wandb:                                                                                                                                                               
            project: opentable                                                                                                                                                 
            result: lm_act_eval-run                                                                                                                                            
          braintrust:                                                                  
            project: multion_opentable                                                                                                                                         
      benchmark:                                                                                                                                                               
        gpt-v:                                                                                                                                                                 
          model: gpt-4-vision-preview                                                  
          max_token: 300                                                                                                                         
          img_fidelity: high                                                                                                                                                   
      metrics:                                                                                                                                                                 
        gpt-v:                                                                         
          inputs:                                                                                                                                
          - GOAL                                                                       
          - QUERY                                                                                                                                                              
          - screenshots                                                                
          args:                                                                        
          - PROMPT_VERSION: multion_trajectory                                         
        html:                                                                                                                                                                  
          inputs:                                                                      
          - HTML                                                                                                                                                               
        llm_relevancy:                                                                 
        - explanation 
  ```