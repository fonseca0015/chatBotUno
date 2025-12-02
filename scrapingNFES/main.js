const puppeteer = require('puppeteer');
const fs = require('fs');
const { Parser } = require('json2csv');

async function scrapeComLogin() {
  console.log('Iniciando o navegador...');
  
  let browser;
  const allScrapedData = [];

  try {
    browser = await puppeteer.launch({
      headless: true, // Pode manter como 'true' agora
      args: ['--start-maximized'],
      defaultViewport: null,
      slowMo: 50 // Adicionado para desacelerar as ações e evitar problemas de carregamento
    });

    const page = await browser.newPage();
    
    // AJUSTE 1: Definir um viewport padrão para estabilizar o modo headless
    await page.setViewport({ width: 1920, height: 1080 });

    // --- ETAPA DE LOGIN ---
    const loginUrl = 'https://unoerpwiki.herokuapp.com/login';
    console.log(`Navegando para a página de login: ${loginUrl}`);
    await page.goto(loginUrl, { waitUntil: 'networkidle2' });
    console.log('Preenchendo credenciais...');
    await page.type('#input-20', 'unolab@unosolucoes.com.br');
    await page.type('#input-22', 'unolab@2025');
    console.log('Realizando o login...');
    await page.click('button.mt-2.text-none.v-btn.v-btn--contained.theme--dark.v-size--large.blue.darken-2');
    await page.waitForNavigation({ waitUntil: 'networkidle2' });
    console.log('Login realizado com sucesso!');
    
    // --- ETAPA DE SCRAPING DA PÁGINA PRINCIPAL ---
    const BASE_URL = 'https://unoerpwiki.herokuapp.com';
    const INITIAL_URL = `${BASE_URL}/pt-br/menu-implantacao/configuracao-plugins/configuracao-nf-e/erros-nfe`;

    console.log(`\nNavegando para a página de extração: ${INITIAL_URL}`);
    await page.goto(INITIAL_URL, { waitUntil: 'networkidle2' });

    console.log('Extraindo palavras-chave da página principal...');
    const keywordsSelector = 'a.v-chip span.teal--text.text--darken-2';
    await page.waitForSelector(keywordsSelector, { timeout: 10000 }); // Aumentei o timeout para 10s por segurança
    const keywords = await page.$$eval(
      keywordsSelector, 
      (spans) => spans.map(span => span.innerText.trim()).filter(text => text.length > 0)
    );
    const keywordsString = keywords.join(', ');
    console.log(`--> Palavras-chave encontradas: ${keywordsString}`);
    console.log('--------------------------------------------------');

    console.log('Extraindo os links das rejeições com "OK"...');
    const linksData = await page.$$eval('p', (paragraphs) => {
        const results = [];
        for (const p of paragraphs) {
            const markOk = p.querySelector('mark.marker-green > strong');
            if (markOk && markOk.innerText.trim() === 'OK') {
                const linkElement = p.querySelector('a');
                if (linkElement) {
                    results.push({
                        href: linkElement.getAttribute('href'),
                        paragraphText: p.innerText.trim().replace(/\s+/g, ' ')
                    });
                }
            }
        }
        return results;
    });

    console.log(`Foram encontrados ${linksData.length} links para visitar.`);
    
    // --- ETAPA DE NAVEGAÇÃO E EXTRAÇÃO DE CONTEÚDO ---
    for (const data of linksData) {
        const cleanHref = data.href.replace(/(\/pt-br)\/\//, '$1/');
        const fullUrl = `${BASE_URL}${cleanHref}`;

        console.log(`\nProcessando: "${data.paragraphText}"`);
        console.log(`--> Navegando para: ${fullUrl}`);
        
        await page.goto(fullUrl, { waitUntil: 'networkidle2' });

        const contentSelector = 'div.contents';
        await page.waitForSelector(contentSelector, { timeout: 5000 });
        const pageContent = await page.evaluate((selector) => {
          const contentArea = document.querySelector(selector);
          return contentArea ? contentArea.innerText : 'ERRO: Bloco de conteúdo não encontrado.';
        }, contentSelector);

        // AJUSTE 2: Adicionar log para verificar o conteúdo extraído
        console.log('--> Conteúdo extraído (amostra):');
        console.log(`${pageContent.substring(0, 400)}...`); // Mostra os primeiros 400 caracteres
        console.log('-----------------------------------');


        allScrapedData.push({
          'Categoria': 'Rejeições NFE',
          'Titulo': data.paragraphText.replace('-OK', '').trim(),
          'Conteudo': pageContent,
          'Palavras chave': keywordsString,
          'Url': fullUrl
        });
    }

    // --- ETAPA FINAL: Geração do arquivo CSV ---
    if (allScrapedData.length > 0) {
      console.log('\n--------------------------------------------------');
      console.log(`Extração finalizada. Total de ${allScrapedData.length} registros coletados.`);
      
      const parser = new Parser();
      const csv = parser.parse(allScrapedData);
      
      fs.writeFileSync('rejeicoes_nfe.csv', '\ufeff' + csv, 'utf-8');
      
      console.log('Arquivo "rejeicoes_nfe.csv" salvo com sucesso!');
    } else {
      console.log('Nenhum dado foi extraído para gerar o arquivo.');
    }

  } catch (error) {
    console.error('Ocorreu um erro durante a execução do script:', error);
  } finally {
    if (browser) {
      console.log('Fechando o navegador.');
      await browser.close();
    }
  }
}

scrapeComLogin();