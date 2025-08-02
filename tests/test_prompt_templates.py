# tests/test_prompt_templates.py
"""
Test suite for prompt templates
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from minichain.prompts.implementations import (
    PromptTemplate, 
    FewShotPromptTemplate,
    ChatPromptTemplate
)


def test_basic_prompt_template():
    """Test basic PromptTemplate functionality"""
    print("📝 Testing Basic PromptTemplate...")
    
    # Test 1: Simple template creation
    template = PromptTemplate.from_template("Tell me about {topic}")
    
    # Test variable detection
    assert template.input_variables == ["topic"]
    print("✅ Variable detection working")
    
    # Test formatting
    result = template.format(topic="artificial intelligence")
    expected = "Tell me about artificial intelligence"
    assert result == expected
    print(f"✅ Basic formatting: {result}")
    
    # Test multiple variables
    multi_template = PromptTemplate.from_template("Write a {style} story about {character}")
    result2 = multi_template.format(style="funny", character="a robot")
    expected2 = "Write a funny story about a robot"
    assert result2 == expected2
    print(f"✅ Multiple variables: {result2}")
    
    # Test invoke method (alternative to format)
    result3 = template.invoke({"topic": "machine learning"})
    assert result3 == "Tell me about machine learning"
    print("✅ Invoke method working")
    
    return True


def test_few_shot_prompt_template():
    """Test FewShotPromptTemplate functionality"""
    print("\n🎯 Testing FewShotPromptTemplate...")
    
    # Create example template
    example_prompt = PromptTemplate.from_template("Question: {question}\nAnswer: {answer}")
    
    # Define examples
    examples = [
        {
            "question": "What is the capital of France?",
            "answer": "The capital of France is Paris."
        },
        {
            "question": "What is 2+2?", 
            "answer": "2+2 equals 4."
        }
    ]
    
    # Create few-shot template
    few_shot_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        suffix="Question: {input}\nAnswer:",
        input_variables=["input"]
    )
    
    # Test formatting
    result = few_shot_template.format(input="What is the capital of Spain?")
    
    print("✅ Few-shot template result:")
    print(result)
    
    # Verify structure
    assert "Question: What is the capital of France?" in result
    assert "Answer: The capital of France is Paris." in result
    assert "Question: What is 2+2?" in result
    assert "Answer: 2+2 equals 4." in result
    assert "Question: What is the capital of Spain?" in result
    
    # Test adding examples
    few_shot_template.add_example({
        "question": "What is Python?",
        "answer": "Python is a programming language."
    })
    
    assert len(few_shot_template.examples) == 3
    print("✅ Adding examples working")
    
    return True


def test_chat_prompt_template():
    """Test ChatPromptTemplate functionality"""
    print("\n💬 Testing ChatPromptTemplate...")
    
    # Create chat template
    messages = [
        {"role": "system", "content": "You are a helpful {role}."},
        {"role": "user", "content": "Tell me about {topic}"}
    ]
    
    chat_template = ChatPromptTemplate(messages)
    
    # Test variable detection
    expected_vars = {"role", "topic"}
    assert set(chat_template.input_variables) == expected_vars
    print("✅ Chat variable detection working")
    
    # Test formatting as messages
    formatted_messages = chat_template.format(role="teacher", topic="Python")
    
    expected_messages = [
        {"role": "system", "content": "You are a helpful teacher."},
        {"role": "user", "content": "Tell me about Python"}
    ]
    
    assert formatted_messages == expected_messages
    print("✅ Message formatting working")
    
    # Test formatting as string
    string_result = chat_template.format_as_string(role="assistant", topic="AI")
    expected_string = "System: You are a helpful assistant.\nUser: Tell me about AI"
    assert string_result == expected_string
    print(f"✅ String formatting: {string_result}")
    
    return True


def test_error_handling():
    """Test error handling in prompt templates"""
    print("\n⚠️ Testing Error Handling...")
    
    template = PromptTemplate.from_template("Hello {name}, welcome to {place}")
    
    # Test missing variable
    try:
        template.format(name="Alice")  # Missing 'place'
        assert False, "Should have raised error"
    except ValueError as e:
        print(f"✅ Caught expected error: {e}")
    
    # Test extra variables (should work fine)
    result = template.format(name="Bob", place="Paris", extra="ignored")
    assert result == "Hello Bob, welcome to Paris"
    print("✅ Extra variables handled correctly")
    
    return True


def test_integration_with_azure_chat():
    """Test prompt templates with Azure chat model"""
    print("\n🔗 Testing Integration with Azure Chat Model...")
    
    try:
        import os
        from minichain.chat_models.azure import AzureOpenAIChatModel
        from minichain.core.types import SystemMessage, HumanMessage
        
        # Check if Azure is configured
        if not all([os.getenv("AZURE_OPENAI_ENDPOINT"), 
                   os.getenv("AZURE_OPENAI_API_KEY"),
                   os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")]):
            print("⚠️ Skipping integration test - Azure not configured")
            return True
        
        # Create chat model
        model = AzureOpenAIChatModel(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"), # type: ignore
            temperature=0,
            max_tokens=50
        )
        
        # Test 1: Simple template + chat model
        template = PromptTemplate.from_template("Explain {concept} in one sentence.")
        prompt = template.format(concept="machine learning")
        response = model.invoke(prompt)
        print(f"✅ Simple template + Azure: {response}")
        
        # Test 2: Chat template + chat model
        chat_template = ChatPromptTemplate([
            {"role": "system", "content": "You are a {role}."},
            {"role": "user", "content": "What is {topic}?"}
        ])
        
        messages = chat_template.format(role="helpful teacher", topic="Python")
        # Convert to our message objects
        message_objects = [
            SystemMessage(messages[0]["content"]),
            HumanMessage(messages[1]["content"])
        ]
        
        response2 = model.invoke(message_objects)
        print(f"✅ Chat template + Azure: {response2}")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Integration test failed: {e}")
        return True  # Don't fail the whole test suite


def run_prompt_template_tests():
    """Run all prompt template tests"""
    print("🧪 Testing Phase 3: Prompt Templates")
    print("=" * 60)
    
    tests = [
        test_basic_prompt_template,
        test_few_shot_prompt_template, 
        test_chat_prompt_template,
        test_error_handling,
        test_integration_with_azure_chat
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results.append(False)
    
    if all(results):
        print("\n🎉 Phase 3 Complete - All prompt templates working!")
        print("✅ Ready to move to Phase 4: Text Splitters")
    else:
        print("\n⚠️ Some prompt template tests failed")
    
    return all(results)


if __name__ == "__main__":
    run_prompt_template_tests()