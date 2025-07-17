#!/usr/bin/env python3
"""
CaptioningVLM 구조 검증 스크립트 (PyTorch 없이 실행 가능)
"""

import os
import ast
import sys

def check_file_exists(filepath):
    """파일 존재 여부 확인"""
    return os.path.exists(filepath)

def parse_python_file(filepath):
    """Python 파일 파싱하여 AST 반환"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return ast.parse(content)
    except Exception as e:
        print(f"파일 파싱 실패 {filepath}: {e}")
        return None

def get_imports_from_ast(tree):
    """AST에서 import 문들 추출"""
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                imports.append(f"{module}.{alias.name}")
    return imports

def get_classes_from_ast(tree):
    """AST에서 클래스 정의들 추출"""
    classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            base_classes = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    base_classes.append(base.id)
                elif isinstance(base, ast.Attribute):
                    base_classes.append(f"{base.value.id}.{base.attr}")
            classes.append({
                'name': node.name,
                'bases': base_classes
            })
    return classes

def get_functions_from_ast(tree):
    """AST에서 함수 정의들 추출"""
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append(node.name)
    return functions

def validate_captionvlm_structure():
    """CaptioningVLM 구조 검증"""
    print("=== CaptioningVLM 구조 검증 ===")
    
    # 필수 파일들 확인
    required_files = [
        'src/models/captionvlm.py',
        'src/models/build.py', 
        'src/models/config.py',
        'src/models/constant.py',
        'train_vlm.py',
        'evaluation.py'
    ]
    
    print("1. 필수 파일 존재 여부:")
    for file in required_files:
        exists = check_file_exists(file)
        print(f"   {'✓' if exists else '✗'} {file}")
        if not exists:
            return False
    
    # CaptioningVLM 클래스 구조 확인
    print("\n2. CaptioningVLM 클래스 구조:")
    captionvlm_tree = parse_python_file('src/models/captionvlm.py')
    if captionvlm_tree:
        classes = get_classes_from_ast(captionvlm_tree)
        captioning_class = next((c for c in classes if c['name'] == 'CaptioningVLM'), None)
        
        if captioning_class:
            print(f"   ✓ CaptioningVLM 클래스 발견")
            print(f"   ✓ 상속 클래스: {captioning_class['bases']}")
            
            if 'CustomVLMModel' in captioning_class['bases']:
                print("   ✓ CustomVLMModel을 올바르게 상속")
            else:
                print("   ✗ CustomVLMModel 상속 문제")
                return False
        else:
            print("   ✗ CaptioningVLM 클래스를 찾을 수 없음")
            return False
    else:
        return False
    
    # 필수 메서드 확인
    print("\n3. CaptioningVLM 필수 메서드:")
    functions = get_functions_from_ast(captionvlm_tree)
    required_methods = [
        '_generate_captions_for_features',
        '_prepare_multimodal_inputs',
        '__init__'
    ]
    
    for method in required_methods:
        if method in functions:
            print(f"   ✓ {method}")
        else:
            print(f"   ✗ {method} 메서드 없음")
            return False
    
    # import 구조 확인
    print("\n4. Import 구조:")
    imports = get_imports_from_ast(captionvlm_tree)
    
    required_imports = [
        'torch',
        'src.models.build.CustomVLMModel'
    ]
    
    for imp in required_imports:
        found = any(imp in i for i in imports)
        print(f"   {'✓' if found else '✗'} {imp}")
        if not found and 'torch' not in imp:  # torch는 runtime에 확인
            return False
    
    return True

def validate_train_integration():
    """훈련 스크립트 통합 검증"""
    print("\n=== 훈련 스크립트 통합 검증 ===")
    
    train_tree = parse_python_file('train_vlm.py')
    if not train_tree:
        return False
    
    imports = get_imports_from_ast(train_tree)
    
    # CaptioningVLM import 확인
    captionvlm_imported = any('CaptioningVLM' in imp for imp in imports)
    print(f"   {'✓' if captionvlm_imported else '✗'} CaptioningVLM import")
    
    # 상수들 import 확인
    constants_imported = any('constant' in imp for imp in imports)
    print(f"   {'✓' if constants_imported else '✗'} 상수 모듈 import")
    
    return captionvlm_imported and constants_imported

def validate_eval_integration():
    """평가 스크립트 통합 검증"""
    print("\n=== 평가 스크립트 통합 검증 ===")
    
    eval_tree = parse_python_file('evaluation.py')
    if not eval_tree:
        return False
    
    # 동적 import 함수 확인
    functions = get_functions_from_ast(eval_tree)
    dynamic_import_exists = 'import_src_modules' in functions
    print(f"   {'✓' if dynamic_import_exists else '✗'} 동적 import 함수")
    
    # CaptioningVLM 처리 로직 확인 (텍스트 검색)
    with open('evaluation.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    captionvlm_handling = 'CaptioningVLM' in content
    fallback_logic = 'CustomVLMModel' in content
    
    print(f"   {'✓' if captionvlm_handling else '✗'} CaptioningVLM 처리 로직")
    print(f"   {'✓' if fallback_logic else '✗'} CustomVLMModel 폴백 로직")
    
    return dynamic_import_exists and captionvlm_handling and fallback_logic

def main():
    """메인 검증 함수"""
    print("CaptioningVLM 구조 검증 시작")
    print("=" * 50)
    
    structure_ok = validate_captionvlm_structure()
    train_ok = validate_train_integration()
    eval_ok = validate_eval_integration()
    
    print("\n" + "=" * 50)
    print("=== 최종 검증 결과 ===")
    print(f"CaptioningVLM 구조: {'✓ 올바름' if structure_ok else '✗ 문제 있음'}")
    print(f"훈련 스크립트 통합: {'✓ 올바름' if train_ok else '✗ 문제 있음'}")
    print(f"평가 스크립트 통합: {'✓ 올바름' if eval_ok else '✗ 문제 있음'}")
    
    overall_ok = structure_ok and train_ok and eval_ok
    print(f"\n전체 통합 상태: {'✓ 성공' if overall_ok else '✗ 실패'}")
    
    if overall_ok:
        print("\n🎉 CaptioningVLM 구조가 올바르게 설정되었습니다!")
        print("   PyTorch 환경에서 정상 작동할 것으로 예상됩니다.")
    else:
        print("\n⚠️  구조에 문제가 있습니다. 위의 오류를 수정하세요.")
    
    return overall_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)