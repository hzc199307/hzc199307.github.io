'use strict';

var fs = require('hexo-fs');
var pathFn = require('path');
var stripIndent = require('strip-indent');
var util = require('hexo-util');
var highlight = util.highlight;

var rCaptionTitleFile = /(.*)?(\s+|^)(\/*\S+)/;
var rLang = /\s*lang:(\w+)/i;

//Add 2016-05-11
var url = require('url');

/**
* Include code tag
*
* Syntax:
*   {% include_code [title] [lang:language] file %}
* file放在与文章同名的路径下
*/

module.exports = function(ctx) {
  //Add 2016-05-11
  var PostAsset = ctx.model('PostAsset');
  
  return function includeCodeTag(args) {
    var config = ctx.config.highlight || {};
    var codeDir = ctx.config.code_dir;
    var arg = args.join(' ');
    var path = '';
    var title = '';
    var lang = '';
    var caption = '';
	
	//Add 2016-05-11
	var asset

    // Add trailing slash to codeDir
    if (codeDir[codeDir.length - 1] !== '/') codeDir += '/';

    if (rLang.test(arg)) {
      arg = arg.replace(rLang, function() {
        lang = arguments[1];
        return '';
      });
    }

    if (rCaptionTitleFile.test(arg)) {
      var match = arg.match(rCaptionTitleFile);
      title = match[1];
	  
	  //Add 2016-05-11
	  var asset = PostAsset.findOne({post: this._id, slug: match[3]});
	  if (!asset) return;
	  
      path = match[3];
    }

    // Exit if path is not defined
    if (!path) return;

	//Modify 2016-05-11 
	//var src = pathFn.join(ctx.source_dir, codeDir, path);
    var src = pathFn.join(ctx.source_dir, "/_posts", url.resolve(ctx.config.root, asset.path));
	//return codeDir + path;

    return fs.exists(src).then(function(exist) {
      if (exist) return fs.readFile(src);
    }).then(function(code) {
      if (!code) return;

      code = stripIndent(code).trim();

      if (!config.enable) {
        return '<pre><code>' + code + '</code></pre>';
      }

      // If the title is not defined, use file name instead
      title = title || pathFn.basename(path);

      // If the language is not defined, use file extension instead
      lang = lang || pathFn.extname(path).substring(1);

	  //Modify 2016-05-11 
	  //caption = '<span>' + title + '</span><a href="' + ctx.config.root + codeDir + path + '">view raw</a>';
      caption = '<span>' + title + '</span><a href="' + url.resolve(ctx.config.root, asset.path) + '">Download code</a>';

      return highlight(code, {
        lang: lang,
        caption: caption,
        gutter: config.line_number,
        tab: config.tab_replace
      });
    });
  };
};
