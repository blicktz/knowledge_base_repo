#!/usr/bin/env python3
"""
DK AI Copywriting Assistant
Generate high-persuasion email copy using RAG and OpenRouter API
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from litellm import completion
from openai import APIError, RateLimitError, APITimeoutError  # For exception handling
from rag_system import SimpleRAG
from prompts import MASTER_PROMPT_TEMPLATE, get_query_for_task

# Load environment variables from .env file
load_dotenv()


class DKCopywriter:
    """Main copywriting assistant class"""
    
    def __init__(self, config_path: str):
        """Initialize the copywriter with configuration"""
        self.config_path = config_path
        self.rag = SimpleRAG(config_path)
        self._setup_litellm()
    
    def _setup_litellm(self):
        """Setup LiteLLM environment variables and configuration"""
        config = self.rag.config
        llm_config = config.get('llm', {}).get('config', {})
        
        api_key = llm_config.get('api_key')
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        
        # Set environment variables for LiteLLM
        os.environ["OPENROUTER_API_KEY"] = api_key
        
        # Set OpenRouter-specific environment variables for proper attribution
        site_url = llm_config.get('site_url', 'https://dk-copywriter.com')
        app_name = llm_config.get('app_name', 'DK AI Copywriting Assistant')
        
        os.environ["OR_SITE_URL"] = site_url
        os.environ["OR_APP_NAME"] = app_name
        
        # Store model and retry configuration
        self.model = llm_config.get('model', 'openrouter/openai/gpt-5')
        self.num_retries = llm_config.get('num_retries', 2)
        self.timeout = llm_config.get('timeout', 30)
        self.fallback_models = llm_config.get('fallback_models', [])
    
    def ensure_knowledge_base(self, documents_dir: str, force_rebuild: bool = False):
        """Ensure the knowledge base is ready"""
        if force_rebuild or not self.rag.check_knowledge_base():
            print("Setting up knowledge base...")
            if not os.path.exists(documents_dir):
                raise FileNotFoundError(f"Documents directory not found: {documents_dir}")
            
            self.rag.add_documents(documents_dir)
            print("✓ Knowledge base ready!")
        else:
            print("✓ Using existing knowledge base")
    
    def generate_copy(self, task: str, debug: bool = False) -> str:
        """Generate copy for the given task"""
        print(f"Generating copy for: {task}")
        print("=" * 60)
        
        # Step 1: Generate contextual query
        query = get_query_for_task(task)
        print(f"Searching knowledge base with query: {query}")
        
        # Step 2: Retrieve relevant context
        context = self.rag.get_context(query, debug=debug)
        num_excerpts = len(context.split('EXCERPT'))-1 if 'EXCERPT' in context else 0
        print(f"Retrieved {num_excerpts} relevant excerpts")
        
        # Step 3: Build master prompt
        master_prompt = MASTER_PROMPT_TEMPLATE.format(
            retrieved_context=context,
            user_task=task
        )
        
        # Step 4: Generate copy using LiteLLM
        print("Generating copy with LLM...")
        
        messages = [
            {
                "role": "system", 
                "content": "You are a world-class direct response copywriter. Generate compelling, action-driving copy that follows proven direct response principles."
            },
            {
                "role": "user", 
                "content": master_prompt
            }
        ]
        
        try:
            if debug:
                print(f"DEBUG: Using model: {self.model}")
                print(f"DEBUG: Retry attempts: {self.num_retries}")
                print(f"DEBUG: Timeout: {self.timeout}s")
                print(f"DEBUG: Fallback models: {self.fallback_models}")
            
            # Primary model attempt with LiteLLM
            response = completion(
                model=self.model,
                messages=messages,
                max_tokens=2000,
                temperature=0.7,
                num_retries=self.num_retries,
                timeout=self.timeout
            )
            
            if debug:
                print(f"DEBUG: LiteLLM response received successfully")
                print(f"DEBUG: Response type: {type(response)}")
                content = response.choices[0].message.content
                print(f"DEBUG: Content length: {len(content)}")
                print(f"DEBUG: Content preview: {content[:200]}...")
            
            return response.choices[0].message.content
            
        except (APIError, RateLimitError, APITimeoutError) as e:
            if debug:
                print(f"DEBUG: Primary model failed: {type(e).__name__}: {str(e)}")
                print(f"DEBUG: Attempting fallback models...")
            
            # Try fallback models
            for fallback_model in self.fallback_models:
                try:
                    if debug:
                        print(f"DEBUG: Trying fallback model: {fallback_model}")
                    
                    response = completion(
                        model=fallback_model,
                        messages=messages,
                        max_tokens=2000,
                        temperature=0.7,
                        num_retries=1,  # Fewer retries for fallbacks
                        timeout=self.timeout
                    )
                    
                    if debug:
                        print(f"DEBUG: Fallback model {fallback_model} succeeded")
                    
                    return f"[Generated using fallback model: {fallback_model}]\n\n{response.choices[0].message.content}"
                    
                except Exception as fallback_error:
                    if debug:
                        print(f"DEBUG: Fallback model {fallback_model} failed: {str(fallback_error)}")
                    continue
            
            # All models failed
            return f"Error: All models failed. Primary error: {str(e)}"
            
        except Exception as e:
            if debug:
                print(f"DEBUG: Unexpected error: {type(e).__name__}: {str(e)}")
                import traceback
                print(f"DEBUG: Traceback: {traceback.format_exc()}")
            return f"Error generating copy: {str(e)}"
    
    def interactive_mode(self, debug: bool = False):
        """Run in interactive mode"""
        print("\n" + "="*60)
        print("DK AI COPYWRITING ASSISTANT")
        print("="*60)
        if debug:
            print("🐛 DEBUG MODE ENABLED")
            print("="*60)
        print("Enter your copywriting tasks. Type 'quit' to exit.\n")
        
        while True:
            try:
                task = input("📝 Describe your email copy task: ").strip()
                
                if task.lower() in ['quit', 'exit', 'q']:
                    print("Thanks for using the DK AI Copywriting Assistant!")
                    break
                
                if not task:
                    print("Please enter a task description.")
                    continue
                
                print()
                copy = self.generate_copy(task, debug=debug)
                
                print("\n" + "="*60)
                print("GENERATED COPY:")
                print("="*60)
                print(copy)
                print("="*60 + "\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="DK AI Copywriting Assistant - Generate high-conversion email copy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_copy.py "Write an email for a webinar about marketing for developers"
  python generate_copy.py --rebuild-kb "Create urgency for limited-time discount"
  python generate_copy.py --interactive
  python generate_copy.py --setup-only
        """
    )
    
    parser.add_argument(
        "task", 
        nargs="?",
        help="The copywriting task to execute"
    )
    
    parser.add_argument(
        "--config", 
        default="openrouter_config.yaml",
        help="Configuration file path (default: openrouter_config.yaml)"
    )
    
    parser.add_argument(
        "--documents-dir",
        default="/Volumes/J15/copy-writing/dk_books_md",
        help="Directory containing DK books in markdown format"
    )
    
    parser.add_argument(
        "--rebuild-kb", 
        action="store_true",
        help="Force rebuild of knowledge base"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--setup-only",
        action="store_true", 
        help="Only setup the knowledge base, don't generate copy"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with detailed similarity scores"
    )
    
    args = parser.parse_args()
    
    # Validate config file exists
    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        # Initialize copywriter
        copywriter = DKCopywriter(str(config_path))
        
        # Setup knowledge base
        copywriter.ensure_knowledge_base(args.documents_dir, args.rebuild_kb)
        
        if args.setup_only:
            print("Knowledge base setup complete!")
            return
        
        if args.interactive:
            copywriter.interactive_mode(debug=args.debug)
        elif args.task:
            copy = copywriter.generate_copy(args.task, debug=args.debug)
            print("\n" + "="*60)
            print("GENERATED COPY:")
            print("="*60)
            print(copy)
            print("="*60)
        else:
            print("Error: No task provided. Use --interactive or specify a task.")
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()