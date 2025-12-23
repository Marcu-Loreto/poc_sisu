# observ.py
import os
from typing import Optional, Any, Dict
from langfuse.decorators import langfuse_context, observe

# =====================================================
# Configuração do Langfuse
# =====================================================
# O Langfuse usa decorators e context managers.
# As variáveis de ambiente são lidas automaticamente:
# - LANGFUSE_PUBLIC_KEY
# - LANGFUSE_SECRET_KEY
# - LANGFUSE_HOST

# =====================================================
# Helpers de Observabilidade
# =====================================================

def get_current_trace_id():
    """
    Retorna o ID do trace atual do contexto do Langfuse.
    """
    try:
        return langfuse_context.get_current_trace_id()
    except:
        return None


def get_current_observation_id():
    """
    Retorna o ID da observação atual do contexto do Langfuse.
    """
    try:
        return langfuse_context.get_current_observation_id()
    except:
        return None


def update_current_trace(output: Any = None, metadata: Optional[Dict] = None):
    """
    Atualiza o trace atual com output e/ou metadata.
    """
    try:
        if output is not None:
            langfuse_context.update_current_trace(output=output)
        if metadata is not None:
            langfuse_context.update_current_trace(metadata=metadata)
    except:
        pass


def update_current_observation(output: Any = None, metadata: Optional[Dict] = None):
    """
    Atualiza a observação atual com output e/ou metadata.
    """
    try:
        if output is not None:
            langfuse_context.update_current_observation(output=output)
        if metadata is not None:
            langfuse_context.update_current_observation(metadata=metadata)
    except:
        pass
