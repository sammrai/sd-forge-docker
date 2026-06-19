# sitecustomize: Python 起動時に自動実行される。
# huggingface_hub 1.0 で削除された HfFolder を復元し、forge の古い gradio
# (from huggingface_hub import HfFolder) が新しい hf_hub でも動くようにする。
# これにより transformers/diffusers(新hf_hub) と forge(旧gradio) を単一envで両立できる。
try:
    import huggingface_hub as _h
    if not hasattr(_h, "HfFolder"):
        try:
            from huggingface_hub import get_token as _get_token
        except Exception:
            _get_token = lambda: None

        class HfFolder:  # noqa: N801
            path_token = None

            @staticmethod
            def get_token():
                try:
                    return _get_token()
                except Exception:
                    return None

            @staticmethod
            def save_token(token=None):
                return None

            @staticmethod
            def delete_token():
                return None

        _h.HfFolder = HfFolder
except Exception:
    pass
