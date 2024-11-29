### 入门例子

#### 安装包

```
pip install open_chatchat
```

#### 示例代码(知识库)

其它参考 python-sdk tests 用例

```
from open_chatchat.chatchat_api import ChatChat
chatchat = ChatChat()
knowledge_base_name='test_kb'
chatchat.knowledge.create_kb(knowledge_base_name="test_kb")
chatchat.knowledge.upload_kb_docs(knowledge_base_name="test_kb",files=[ "data/upload_file2.txt"],))
chatchat.knowledge.search_kb_docs(knowledge_base_name=knowledge_base_name, query="hello")    
```

```
chatchat = ChatChat()
knowledge_base_name='test_kb'
chatchat.knowledge.create_kb(knowledge_base_name="test_kb")
chatchat.knowledge.upload_kb_docs(knowledge_base_name="test_kb",files=[ "data/upload_file2.txt"],))
chatchat.knowledge.search_kb_docs(knowledge_base_name=knowledge_base_name, query="hello")    
```

### 支持标准的open ai接口

默认启动api地址为 http://127.0.0.1:7861/

##### 使用时用openai安装包

pip install openai

##### 支持接口如下

/v1/models
/v1/chat/completions
/v1/completions
/v1/embeddings
/v1/images/generations
/v1/images/variations
/v1/images/edit
/v1/audio/translations
/v1/audio/transcriptions
/v1/audio/speech
/v1/files
/v1/files
/v1/files/{file_id}
/v1/files/{file_id}/content
/v1/files/{file_id}
