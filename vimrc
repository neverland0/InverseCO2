call plug#begin()
" The default plugin directory will be as follows:
"   - Vim (Linux/macOS): '~/.vim/plugged'
"   - Vim (Windows): '~/vimfiles/plugged'
"   - Neovim (Linux/macOS/Windows): stdpath('data') . '/plugged'
" You can specify a custom plugin directory by passing it as the argument
"   - e.g. `call plug#begin('~/.vim/plugged')`
"   - Avoid using standard Vim directory names like 'plugin'

" Make sure you use single quotes

" Shorthand notation; fetches https://github.com/junegunn/vim-easy-align
Plug 'morhetz/gruvbox'
Plug 'chriskempson/base16-vim'
Plug 'lervag/vimtex'
let g:vimtex_compiler_latexmk_engines = {
    \ '_'                : '-xelatex',
    \}
Plug 'SirVer/ultisnips'
Plug 'honza/vim-snippets'
let g:UltiSnipsExpandTrigger       = '<Tab>'    " use Tab to expand snippets
let g:UltiSnipsJumpForwardTrigger  = '<Tab>'    " use Tab to move forward through tabstops
let g:UltiSnipsJumpBackwardTrigger = '<S-Tab>'  " use Shift-Tab to move backward through tabstops
let g:UltiSnipsSnippetDirectories=["UltiSnips", "MySnippets"]

call plug#end()
" You can revert the settings after the call like so:
"   filetype indent off   " Disable file-type-specific indentation
"   syntax off            " Disable syntax highlighting
autocmd vimenter * ++nested colorscheme gruvbox
set bg=dark
set nu
set so=999 "c-y c-e 滚屏时光标随屏幕滚动
set hlsearch
" Uncomment the following to have Vim jump to the last position when
" reopening a file
if has("autocmd")
  au BufReadPost * if line("'\"") > 1 && line("'\"") <= line("$") | exe "normal! g'\"" | endif
endif
filetype plugin on
set encoding=utf-8
let mapleader = " "
let maplocalleader="\<space>"
noremap ww :w<CR>
noremap <leader>q :q!<CR>
noremap <leader>w :w<CR>
noremap <leader>z :wq<CR>
noremap <leader><Tab> <C-W><C-W>
noremap <Space>c :write<CR>:VimtexCompile<CR>
noremap <leader>s :call UltiSnips#RefreshSnippets()<CR>

set ts=4
set expandtab
set autoindent
