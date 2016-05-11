// Add a tail to every post from tail.md
// Great for adding copyright info

var fs = require('fs');

hexo.extend.filter.register('before_post_render', function(data){
    if(data.copyright == false) return data;
    var file_content = fs.readFileSync('themes\\jacman\\copyright.md');
    if(file_content && data.content.length > 50) 
    {
		//data.content += "<hr>";
        data.content += file_content;
        var permalink = '\n本文链接：' + data.permalink;
        data.content += permalink;
		var githublink = '\n永久链接：http://hzc199307.github.io/' + data.path;
		data.content += githublink;
    }
    return data;
});