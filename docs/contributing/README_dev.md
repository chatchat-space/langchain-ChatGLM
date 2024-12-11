# Langchain-Chatchat 源代码部署/开发部署指南



# 目录

[toc]

## 0. 拉取项目代码

如果您是想要使用源码启动的用户，请直接拉取 master 分支代码
基于源码部署完成后可以单步调试和验证，达到快速理解源码的效果。
此手册是给到开发人员参考的，非专业开发人员建议使用docker的方式安装LangChain-Chatchat。
```shell
mkdir -p ~/project/Langchain-Chatchat
cd ~/project/Langchain-Chatchat
git clone https://github.com/chatchat-space/Langchain-Chatchat.git
```

## 1. 初始化开发环境

Langchain-Chatchat 自 0.3.0 版本起，为方便支持用户使用 pip 方式安装部署，以及为避免环境中依赖包版本冲突等问题，
在源代码/开发部署中不再继续使用 requirements.txt 管理项目依赖库，转为使用 Poetry 进行环境管理。

### 1.1 安装 Poetry

> 在安装 Poetry 之前，如果您使用 Conda，请创建并激活一个新的 Conda 环境，例如使用 `conda create -n poetry python=3.11` 创建一个新的 Conda 环境。

```sh
conda create -n poetry  python=3.11
conda activate poetry
```

安装 Poetry: [Poetry 安装文档](https://python-poetry.org/docs/#installing-with-pipx)

> [!Note]
> 如果你没有其它 poetry 进行环境/依赖管理的项目，利用 pipx 或 pip 都可以完成 poetry 的安装，

> [!Note]
> 如果您使用 Conda 或 Pyenv 作为您的环境/包管理器，在安装Poetry之后，
> 使用如下命令使 Poetry 使用 virtualenv python environment (`poetry config virtualenvs.prefer-active-python true`)

### 创建Langchain-Chatchat的python环境

conda create -n lccc python=3.11 

conda activate lccc

### 1.2 安装源代码/开发部署所需依赖库

编辑项目的配置文件pyproject.toml，将poetry的源为清华源，要不然下载会非常慢

```shell
cd ~/project/Langchain-Chatchat/libs/chatchat-server/
nano pyproject.toml
```

在pyproject.toml文件内部找到下面被屏蔽的语句，去掉下面语句前面的屏蔽符号 #

```json
[[tool.poetry.source]]
name = "tsinghua"
url = "https://pypi.tuna.tsinghua.edu.cn/simple/"
priority = "primary"
```

进入主项目目录，并安装 Langchain-Chatchat 依赖

```shell
cd  ~/project/Langchain-Chatchat/libs/chatchat-server/
poetry install --with lint,test -E xinference

# or use pip to install in editing mode:
pip install -e .
```

> [!Note]
> Poetry install 后会在你的虚拟环境中 site-packages 路径下生成一个 chatchat-`<version>`.dist-info 文件夹带有 direct_url.json 文件，这个文件指向你的开发环境 

### 1.3 更新开发部署环境依赖库

当开发环境中所需的依赖库发生变化时，一般按照更新主项目目录(`Langchain-Chatchat/libs/chatchat-server/`)下的 pyproject.toml 再进行 poetry update 的顺序执行。

### 1.4 将更新后的代码打包测试

如果需要对开发环境中代码打包成 Python 库并进行测试，可在主项目目录执行以下命令：

```shell
cd ~/project/Langchain-Chatchat/libs/chatchat-server
poetry build
```

命令执行完成后，在主项目目录下会新增 `dist` 路径，其中存储了打包后的 Python 库。

```shell
cd ~/project/Langchain-Chatchat/libs/chatchat-server/dist
ls
```



#### 使用源代码安装模式

你可以使用 `pip` 的 `-e` 选项（可编辑模式）来安装源代码。这种模式下，`pip` 会在你的项目目录中创建一个符号链接，而不是复制文件。这样，你对源代码的任何修改都会立即生效。步骤如下：

1. 确保你在项目的根目录下（即包含 `pyproject.toml` 文件的目录）。
2. 使用以下命令进行可编辑模式安装：

```sh
cd ~/project/Langchain-Chatchat/libs/chatchat-server/
pip install -e .
```

这个命令会在你的项目目录中创建一个符号链接，并将包安装到你的 Python 环境中。

**可编辑模式安装**：使用 `pip install -e .` 进行可编辑模式安装，确保你在调试过程中对源代码进行修改时，这些修改能够立即生效。

**单一源码**：可编辑模式安装不会复制源代码文件，而是创建符号链接，确保你只有一份源代码。

通过这种方式，你可以方便地进行调试和开发，而不需要担心源代码的重复和同步问题。



## 2. 设置源代码根目录

如果您在开发时所使用的 IDE 需要指定项目源代码根目录，请将主项目目录(`Langchain-Chatchat/libs/chatchat-server/`)设置为源代码根目录。

执行以下命令之前，请先设置当前目录和项目数据目录：

```sh
# 临时设置环境变量
export CHATCHAT_ROOT=~/project/Langchain-Chatchat/libs/chatchat-server/chatchat/data

# 永久设置环境变量
nano ~/.bashrc
# 在.bashrc最后添加下面命令
export CHATCHAT_ROOT=~/project/Langchain-Chatchat/libs/chatchat-server/chatchat/data
#   使配置生效    
source ~/.bashrc
```

## 3. 关于 chatchat 配置项

### ollama获取当前模型列表(可选)

ollama的安装和配置请参考ollama的官方说明，下面的命令是在ollama和大模型配置好的情况下获取可用的模型列表，方便在下一步配置中使用。

如果当前已安装的模型不能满足项目要求，可以从 https://ollama.com/library 查询是否有适合的模型。

```sh
# 查看ollama版本 
ollama --version

# 查看ollama下已经有的模型列表
ollama list
```



### xinference配置(必须)

现有源码在下一步生成默认配置文件时依赖xinference的服务。

```sh
# 建立xinference的环境
conda create -n xinference python=3.11 
conda activate xinference

# 如果后续准备使用ollama框架访问大模型，这里可以简化配置xinference的，只执行"xinference[transformers]"即可。
pip install "xinference[transformers]"

# 如果后续准备使用xinference框架访问大模型，则需要再执行安装后去下载并配置大模型，详情请见xinference框架说明
# TODO 

# 启动xinference服务
xinference-local --host 0.0.0.0 --port 9997
```

## 3. 关于 chatchat 配置项

从 `0.3.1` 版本开始，所有配置项改为 `yaml` 文件，具体参考 [Settings](settings.md)。

执行以下命令初始化项目配置文件和数据目录：
```shell
# 生成默认配置文件
cd ~/project/Langchain-Chatchat/libs/chatchat-server/chatchat
python cli.py init
# 编辑配置文件
cd ~/project/Langchain-Chatchat/libs/chatchat-server/chatchat/data
# 根据当前使用的框架和模型修改配置
nano model_settings.yaml 
nano basic_settings.yaml 
```

## 4. 初始化知识库

### 配置 nltk_data

当机器能够科学上网时可以跳过这个步骤，这个步骤相当于离线配置nltk。

可以从 https://github.com/nltk/nltk_data/tree/gh-pages下载最新版本，参考[安装和使用nltk_nltk国内镜像源-CSDN博客](https://blog.csdn.net/make_progress/article/details/116941669) 进行设置。

### 初始化知识库

> [!WARNING]
> 这个命令会清空数据库、删除已有的配置文件，如果您有重要数据，请备份。

```shell
cd ~/project/Langchain-Chatchat/libs/chatchat-server/chatchat
python cli.py kb --recreate-vs
```
如需使用其它 Embedding 模型，或者重建特定的知识库，请查看 `python cli.py kb --help` 了解更多的参数。

## 5. 启动服务

```shell
cd ~/project/Langchain-Chatchat/libs/chatchat-server/chatchat
python cli.py start -a
```

如需调用 API，请参考 [API 使用说明](api.md)
