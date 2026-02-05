#!/bin/bash

# 配置文件路径
CONFIG_FILE="/etc/nginx/http.d/default.conf" 

direct_link=$(cat /opt/alist/alist | grep -ao "\\\$\\{[a-z]*\\}\\\$\\{[a-z]*\\}\\\$\\{[a-z]*\\}" | sed "s/\\\$/\$dollar/g")
direct_link_sign=$direct_link

sign_cond=$(cat /opt/alist/alist | grep -ao '[a-z]*!=="preview"&&[a-z]*\.sign' | sed "s/\\\$/\$dollar/g")
sign_str=$(cat /opt/alist/alist | grep -ao '\?sign=\$\{[a-z]*\.sign\}' | sed "s/\\\$/\$dollar/g")

#这里有两个匹配的，第二处不知道什么应用场景，仅修改第一处。
login_cmd=$(cat /opt/alist/alist | grep -ao '[a-zA-Z]\.success.*?login\.success.*?,.*?.token\),' | head -n1 | sed "s/\\\$/\$dollar/g")
login_cmd_sign=$login_cmd
post_api_auth=$(cat /opt/alist/alist | grep -ao '[a-zA-Z]*\.post\("/auth/login"' | awk -F'(' '{print $1}' | head -n1)

location_getsignmd5=$(cat "$CONFIG_FILE" | grep "cgi-bin/soutv" | sed 's/^\s*//g' | head -n1)

if [ ! -f /data/nosign.txt ] && [ -f /data/guestpass.txt ] && [ -f /data/guestlogin.txt ]; then
    
    if [ -f /data/salt.txt ]; then
        sign=$(cat /data/salt.txt | tr -d '\r\n' | md5sum | awk '{print $1}')
    else
        sign=$({ ip link show; date; } | tr -d '\r\n' | md5sum | awk '{print $1}')
    fi
    
    echo -n "$sign" >/index/md5

    #给URL加上签名
    js='
(function(){
    return "?sign=" + localStorage.getItem("signmd5");
})();
'
    js=$(echo -n "$js" | base64 -w 0)
    direct_link_sign="$direct_link\$dollar{(()=>{const encodedFunc = \"$js\"; const decodedFunc = atob(encodedFunc); return eval(decodedFunc);})()}"

    #登录的同时远程获取md5
    js='
(function(){
    '"$post_api_auth"'("/getsignmd5", "cat md5", {
  headers: {
    "Content-Type": "text/plain"
  }
})
        .then(response => {
            localStorage.setItem("signmd5", response);
        });
})();
'
    js=$(echo -n "$js" | base64 -w 0)
    login_cmd_sign="$login_cmd(()=>{const encodedFunc = \"$js\"; const decodedFunc = atob(encodedFunc); eval(decodedFunc);})(),"

fi

alist_address="http://127.0.0.1:$(cat "$CONFIG_FILE" | grep -o ':[0-9]*/d/' | tr -d ':' | awk -F'/' '{print $1}'| head -n1)"
docker_ip="127.0.0.1"
if [ "$alist_address"x = "http://127.0.0.1:5244"x ]; then
    docker_ip=$(ifconfig "eth0" | grep "inet addr" | grep -o "([0-9]{1,3}\.){3}[0-9]{1,3}" | head -n1)
fi

sed -i "s/sign=[^']*/sign=$sign/g" /etc/nginx/http.d/emby.js

gen_api_get_list_sub_filter() {
    while read -r line; do
        post_api=$(echo "$line" | awk -F'(' '{print $1}')
        js='
(function(){
    '"$post_api"'("/getsignmd5", "cat md5", {
  headers: {
    "Content-Type": "text/plain"
  }
})
        .then(response => {
            localStorage.setItem("signmd5", response);
        });
})();
'
        js=$(echo -n "$js" | base64 -w 0)
        sub_line="(()=>{(()=>{const encodedFunc = \"$js\"; const decodedFunc = atob(encodedFunc); eval(decodedFunc);})();return $line})()"
        echo 'sub_filter '"'$line'"' '"'$sub_line'"';'
    done < <(
        cat /opt/alist/alist | grep -ao '[a-zA-Z]*\.post\("/fs/get".*?\)'
        cat /opt/alist/alist | grep -ao '[a-zA-Z]*\.post\("/fs/list".*?\)'
        )
}

config_location() {
    sign=$1
    if [ -z "$sign" ]; then
        NEW_CONFIG=''
    else
        NEW_CONFIG='
                if ($modified_uri ~ ^/d/[^\\?]*(?!.*sign='"$sign"'($|&))) {
                    return 302 http://'"$docker_ip"':45678$request_uri;
                }
'
    fi

    # 临时文件
    TMP_FILE=$(mktemp)

    # 检查文件是否存在
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Nginx configuration file not found at $CONFIG_FILE"
        exit 1
    fi

    # 处理文件
    awk -v new_config="$NEW_CONFIG" '
    BEGIN {
        in_target_block = 0
    }

    # 匹配到目标location行时立即插入配置
    /location \/d\/ \{/ {
        print
        # 立即插入新配置
        printf("%s",new_config)
        in_target_block = 1
        next
    }

    # 在目标块中检查现有的if (sign=)条件
    in_target_block && /if.*sign=/ {
        # 跳过现有的if块（不打印）
        while (getline > 0) {
            if (/}[[:space:]]*$/) {
                break
            }
        }
	in_target_block = 0
        next
    }

    # 其他情况直接打印
    {
        print
    }
    ' "$CONFIG_FILE" | sed '/^[[:space:]]*$/N; /^\n$/D' > "$TMP_FILE"

    mv -f "$TMP_FILE" "$CONFIG_FILE"

}

config_geo() {
    sign=$1
    if [ -z "$sign" ]; then
        NEW_CONFIG=''
    else
        NEW_CONFIG='
geo $remote_addr $is_external {
    default 1;

}
'
    fi

    # 临时文件
    TMP_FILE=$(mktemp)

    # 检查文件是否存在
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Nginx configuration file not found at $CONFIG_FILE"
        exit 1
    fi

    # 处理文件
    awk -v new_config="$NEW_CONFIG" '
    BEGIN {
        print
        # 立即插入新配置
        printf("%s",new_config)
    }

    # 在目标块中检查现有的geo块
    /geo \$remote_addr \$is_external/ {
        # 跳过现有的块（不打印）
        while (getline > 0) {
            if (/}[[:space:]]*$/) {
                break
            }
        }
        next
    }

    # 其他情况直接打印
    {
        print
    }
    ' "$CONFIG_FILE" | sed '/^[[:space:]]*$/N; /^\n$/D' > "$TMP_FILE"

    mv -f "$TMP_FILE" "$CONFIG_FILE"

}

config_uri_map() {
    sign=$1
    if [ -z "$sign" ]; then
        NEW_CONFIG=''
    else
        NEW_CONFIG='
map $is_external $modified_uri {
    default         $uri?sign=$arg_sign;
    0               $uri?sign='"$sign"';
    1               $uri?sign=$arg_sign;
}
'
    fi

    # 临时文件
    TMP_FILE=$(mktemp)

    # 检查文件是否存在
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Nginx configuration file not found at $CONFIG_FILE"
        exit 1
    fi

    # 处理文件
    awk -v new_config="$NEW_CONFIG" '
    BEGIN {
        print
        # 立即插入新配置
        printf("%s",new_config)
    }

    # 在目标块中检查现有的map块
    /map \$is_external \$modified_uri/ {
        # 跳过现有的块（不打印）
        while (getline > 0) {
            if (/}[[:space:]]*$/) {
                break
            }
        }
        next
    }

    # 其他情况直接打印
    {
        print
    }
    ' "$CONFIG_FILE" | sed '/^[[:space:]]*$/N; /^\n$/D' > "$TMP_FILE"

    mv -f "$TMP_FILE" "$CONFIG_FILE"

}

config_geo_dollar() {
    sign=$1
    if [ -z "$sign" ]; then
        NEW_CONFIG=''
    else
        NEW_CONFIG='
geo $dollar {
    default "$";
}
'
    fi

    # 临时文件
    TMP_FILE=$(mktemp)

    # 检查文件是否存在
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Nginx configuration file not found at $CONFIG_FILE"
        exit 1
    fi

    # 处理文件
    awk -v new_config="$NEW_CONFIG" '
    BEGIN {
        print
        # 立即插入新配置
        printf("%s",new_config)
    }

    # 在目标块中检查现有的geo块
    /geo \$dollar/ {
        # 跳过现有的块（不打印）
        while (getline > 0) {
            if (/}[[:space:]]*$/) {
                break
            }
        }
        next
    }

    # 其他情况直接打印
    {
        print
    }
    ' "$CONFIG_FILE" | sed '/^[[:space:]]*$/N; /^\n$/D' > "$TMP_FILE"

    mv -f "$TMP_FILE" "$CONFIG_FILE"

}

config_location_assets() {
    sign=$1
    if [ -z "$sign" ]; then
        NEW_CONFIG=''
    else
        NEW_CONFIG='
    location /assets {
        proxy_pass '"$alist_address"';
        proxy_set_header Accept-Encoding "";
        sub_filter '"'$direct_link'"' '"'$direct_link_sign'"';
        sub_filter '"'$login_cmd'"' '"'$login_cmd_sign'"';
        sub_filter '"'$sign_str'"' '"''"';
        sub_filter '"'$sign_cond'"' '"'false'"';
        '"$(gen_api_get_list_sub_filter)"'
        sub_filter_once off;
        sub_filter_types *;
        proxy_cache apicache;
    }
'
    fi

    # 临时文件
    TMP_FILE=$(mktemp)

    # 检查文件是否存在
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Nginx configuration file not found at $CONFIG_FILE"
        exit 1
    fi

    # 处理文件
    awk -v new_config="$NEW_CONFIG" '
    BEGIN {
        in_target_block = 0
    }

    # 匹配到目标server行时立即插入配置
    /server \{/ {
        print
        # 立即插入新配置
        printf("%s",new_config)
        in_target_block = 1
        next
    }

    # 在目标块中检查现有的location /assets
    in_target_block && /location \/assets/ {
        # 跳过现有的if块（不打印）
        while (getline > 0) {
            if (/}[[:space:]]*$/) {
                break
            }
        }
	in_target_block = 0
        next
    }

    # 其他情况直接打印
    {
        print
    }
    ' "$CONFIG_FILE" | sed '/^[[:space:]]*$/N; /^\n$/D' > "$TMP_FILE"

    mv -f "$TMP_FILE" "$CONFIG_FILE"

}

config_location_getsignmd5() {
    sign=$1
    if [ -z "$sign" ]; then
        NEW_CONFIG=''
    else
        NEW_CONFIG='
    location /api/getsignmd5 {
        '"$location_getsignmd5"'
    }
'
    fi

    # 临时文件
    TMP_FILE=$(mktemp)

    # 检查文件是否存在
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Nginx configuration file not found at $CONFIG_FILE"
        exit 1
    fi

    # 处理文件
    awk -v new_config="$NEW_CONFIG" '
    BEGIN {
        in_target_block = 0
    }

    # 匹配到目标server行时立即插入配置
    /server \{/ {
        print
        # 立即插入新配置
        printf("%s",new_config)
        in_target_block = 1
        next
    }

    # 在目标块中检查现有的location /assets
    in_target_block && /location \/api\/getsignmd5/ {
        # 跳过现有的if块（不打印）
        while (getline > 0) {
            if (/}[[:space:]]*$/) {
                break
            }
        }
	in_target_block = 0
        next
    }

    # 其他情况直接打印
    {
        print
    }
    ' "$CONFIG_FILE" | sed '/^[[:space:]]*$/N; /^\n$/D' > "$TMP_FILE"

    mv -f "$TMP_FILE" "$CONFIG_FILE"

}

config_emby_local_direct_link_server() {
    sign=$1
    if [ -z "$sign" ]; then
        NEW_CONFIG=''
    else
        NEW_CONFIG='
server  {
    listen '"$docker_ip"':45678;
    server_name emby_local_direct_link;
    location  /d/ {
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Host $http_host;
        proxy_set_header Host $http_host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header Range $http_range;
        proxy_set_header If-Range $http_if_range;
        proxy_redirect off;
        proxy_pass '"$alist_address"'/d/;
        limit_req zone=two burst=2;
    }
}
'
    fi

    # 临时文件
    TMP_FILE=$(mktemp)

    # 检查文件是否存在
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Nginx configuration file not found at $CONFIG_FILE"
        exit 1
    fi

    # 处理文件
    awk -v new_config="$NEW_CONFIG" '
    BEGIN {
        print
        # 立即插入新配置
        printf("%s",new_config)
        block_num = 0
    }

    # 在目标块中检查现有的server块
    /server  \{/ {
        # 跳过现有的块（不打印）
        while (getline > 0) {
            if (/}[[:space:]]*$/) {
                block_num = block_num + 1
            }
            if (block_num == 2) {
                block_num = 0
                break
            }
        }
        next
    }

    # 其他情况直接打印
    {
        print
    }
    ' "$CONFIG_FILE" | sed '/^[[:space:]]*$/N; /^\n$/D' > "$TMP_FILE"

    mv -f "$TMP_FILE" "$CONFIG_FILE"
}

config_block_p() {
    NEW_CONFIG='
    location ^~ /p/ {
        deny all;
        return 403;
    }
'

    # 临时文件
    TMP_FILE=$(mktemp)

    # 检查文件是否存在
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Nginx configuration file not found at $CONFIG_FILE"
        exit 1
    fi

    # 处理文件
    awk -v new_config="$NEW_CONFIG" '
    BEGIN {
        in_target_block = 0
    }

    # 匹配到目标server行时立即插入配置
    /server \{/ {
        print
        # 立即插入新配置
        printf("%s",new_config)
        in_target_block = 1
        next
    }

    # 在目标块中检查现有的location
    in_target_block && /location \^\~ \/p\/ \{/ {
        # 跳过现有的if块（不打印）
        while (getline > 0) {
            if (/}[[:space:]]*$/) {
                break
            }
        }
	in_target_block = 0
        next
    }

    # 其他情况直接打印
    {
        print
    }
    ' "$CONFIG_FILE" | sed '/^[[:space:]]*$/N; /^\n$/D' > "$TMP_FILE"

    mv -f "$TMP_FILE" "$CONFIG_FILE"
}

config_strm() {
    sign=$1
    NEW_CONFIG='
    location ^~ /dav/strm {
        proxy_pass '"$alist_address"'/dav/strm;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Host $http_host;
        proxy_set_header Host $http_host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header Range $http_range;
        proxy_set_header If-Range $http_if_range;
        proxy_set_header Accept-Encoding "";
        proxy_redirect off;
        proxy_cache apicache;
        sub_filter_once off;
        sub_filter http://xiaoya.host:5678 http://$http_host;
        sub_filter SIGN_STR '"$sign"';
        sub_filter_types text/plain;
    }
'

    # 临时文件
    TMP_FILE=$(mktemp)

    # 检查文件是否存在
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Nginx configuration file not found at $CONFIG_FILE"
        exit 1
    fi

    # 处理文件
    awk -v new_config="$NEW_CONFIG" '
    BEGIN {
        in_target_block = 0
    }

    # 匹配到目标server行时立即插入配置
    /server \{/ {
        print
        # 立即插入新配置
        printf("%s",new_config)
        in_target_block = 1
        next
    }

    # 在目标块中检查现有的location
    in_target_block && /location \^\~ \/dav\/strm \{/ {
        # 跳过现有的if块（不打印）
        while (getline > 0) {
            if (/}[[:space:]]*$/) {
                break
            }
        }
	in_target_block = 0
        next
    }

    # 其他情况直接打印
    {
        print
    }
    ' "$CONFIG_FILE" | sed '/^[[:space:]]*$/N; /^\n$/D' > "$TMP_FILE"

    mv -f "$TMP_FILE" "$CONFIG_FILE"
}

config_location "$sign"
config_uri_map "$sign"
config_geo "$sign"

#给网页直链加上签名
config_geo_dollar "$sign"
config_location_assets "$sign"

#给nginx放开获取signmd5的api白名单
config_location_getsignmd5 "$sign"

#增加emby本地直链服务器
config_emby_local_direct_link_server "$sign"

#不允许访问/p/，没有带签名匿名访问有安全隐患
config_block_p

#给strm文件内容加上签名
config_strm "$sign"

if [ -f /run/nginx/nginx.pid ]; then
    nginx -s reload
fi
