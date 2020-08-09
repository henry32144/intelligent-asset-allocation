import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Grid from '@material-ui/core/Grid';
import Button from '@material-ui/core/Button';
import Typography from '@material-ui/core/Typography';
import CreateOutlinedIcon from '@material-ui/icons/CreateOutlined';
import MoneyOffOutlinedIcon from '@material-ui/icons/MoneyOffOutlined';
import ShowChartOutlinedIcon from '@material-ui/icons/ShowChartOutlined';
import Box from '@material-ui/core/Box';
import { Line } from 'react-chartjs-2';
import { COLOR_PALETTES, PERFORMANCE_SHOWCASE } from '../Constants';
import Card from '@material-ui/core/Card';
import CardActions from '@material-ui/core/CardActions';
import CardContent from '@material-ui/core/CardContent';
import CardHeader from '@material-ui/core/CardHeader';
import StarIcon from '@material-ui/icons/StarBorder';
import Link from '@material-ui/core/Link';
import Container from '@material-ui/core/Container';

const useStyles = makeStyles((theme) => ({
  masthead: {
    background: "url(../static/landing_img2.jpg)",
    backgroundPosition: "center",
    backgroundRepeat: "no-repeat",
    backgroundSize: "cover",
    marginLeft: "auto",
    marginRight: "auto",
    height: "100vh",
    maxHeight: "1600px",
    padding: theme.spacing(0, 2, 0),
  },
  landingSectionLight: {
    backgroundColor: "#FFF",
    marginLeft: "auto",
    marginRight: "auto",
    padding: theme.spacing(15, 2, 30),
  },
  landingSectionDark: {
    backgroundColor: "#E5E5E5",
    marginLeft: "auto",
    marginRight: "auto",
    padding: theme.spacing(15, 2, 30),
  },
  mastheadText: {
    color: "white"
  },
  mastheadButton: {
    backgroundColor: "#00873c",
    color: "white"
  },
  icon: {
    height: "64px",
    width: "64px"
  },
  iconContainer: {
    textAlign: "center"
  },
  featureTitle: {
    textAlign: "center",
    margin: theme.spacing(5, 0, 5),
  },
  sectionTitle: {
    textAlign: "center",
    margin: theme.spacing(-5, 0, 10),
  },
  heroContent: {
    padding: theme.spacing(8, 0, 6),
  },
  cardHeader: {
    borderBottom: "1px solid rgba(0, 0, 0, 0.12)",
  },
  cardPricing: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'baseline',
    marginBottom: theme.spacing(2),
  },
  '@global': {
    ul: {
      margin: 0,
      padding: 0,
      listStyle: 'none',
    },
  },
  link: {
    margin: theme.spacing(1, 1.5),
  },
  footer: {
    paddingTop: theme.spacing(1),
    paddingBottom: theme.spacing(1),
    [theme.breakpoints.up('sm')]: {
      paddingTop: theme.spacing(3),
      paddingBottom: theme.spacing(3),
    },
  },
}));

function Copyright() {
  return (
    <div>
    <Typography variant="body2" color="textSecondary" align="center">
      {'Copyright © '}
      <Link color="inherit" href="https://material-ui.com/">
        HuggingMoney
      </Link>{' '}
      {new Date().getFullYear()}
      {'.'}
    </Typography>
    <Typography variant="body2" color="textSecondary" align="center">
    {'Masthead Photo by '}
    <Link color="inherit" href="https://unsplash.com/@austindistel?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">
      Austin Distel
    </Link>
    {' on'}
    <Link color="inherit" href="https://unsplash.com/s/photos/finance?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">
      Unsplash
    </Link>{' '}
  </Typography>
  </div>
  );
}

export default function LandingPage(props) {
  const classes = useStyles();
  const [portfolioPerformances, setPortfolioPerformance] = React.useState({});

  const startOnClick = () => {
    if (props.userData.userId == undefined) {
      window.location.pathname = './login'
    } else {
      window.location.pathname = './dashboard'
    }
  }

  const generateDataTemplate = (index) => {
    return {
      label: '',
      fill: false,
      lineTension: 0.1,
      backgroundColor: COLOR_PALETTES[index],
      borderColor: COLOR_PALETTES[index],
      borderCapStyle: 'butt',
      borderDash: [],
      borderDashOffset: 0.0,
      borderJoinStyle: 'miter',
      pointBorderColor: COLOR_PALETTES[index],
      pointBackgroundColor: '#fff',
      pointBorderWidth: 0,
      pointHoverRadius: 5,
      pointHoverBackgroundColor: COLOR_PALETTES[index],
      pointHoverBorderColor: COLOR_PALETTES[index],
      pointHoverBorderWidth: 2,
      pointRadius: 0,
      pointHitRadius: 5,
      data: []
    }
  }

  const setDemoDataset = () => {
    var performance = PERFORMANCE_SHOWCASE.data.all_values;
    var SP500 = PERFORMANCE_SHOWCASE.data.SP500;
    var date = PERFORMANCE_SHOWCASE.data.date;

    var maxVal = 42;
    var delta = Math.floor(date.length / maxVal);
    var lesserDate = [];
    var lesserData = [];
    var lesserSP500Data = [];
    var dataset = generateDataTemplate(0);
    var SP500Dataset = generateDataTemplate(1);
    const originalData = performance.slice(1);
    const originalSP500Data = SP500.slice(1);

    for (var i = 0; i < performance.length; i = i + delta) {
      lesserData.push(originalData[i]);
      lesserSP500Data.push(originalSP500Data[i]);
      lesserDate.push(date[i]);
    }

    dataset.label = "History performance";
    dataset.data = lesserData;
    SP500Dataset.label = "SP500 Index";
    SP500Dataset.data = lesserSP500Data;

    var newPerformanceData = {
      labels: lesserDate,
      datasets: [dataset, SP500Dataset]
    }

    console.log(newPerformanceData);
    console.log(originalData[originalData.length - 1]);
    //setPortfolioPerformance(newPerformanceData);
    return newPerformanceData;
  }

  const tiers = [
    {
      title: 'Free',
      price: '0',
      description: ['1 month trial', '2 basic models', '1 custom portfolio', '15 companys in a portfolio'],
      buttonText: 'Sign up for free',
      buttonVariant: 'outlined',
    },
    {
      title: 'Pro',
      subheader: 'Most popular',
      price: '15',
      description: ['all basic models', 'cutting edge models', '99 custom portfolios', 'unlimited companys in a portfolio'],
      buttonText: 'Get started',
      buttonVariant: 'contained',
    },
  ];
  return (
    <div>
      {/* Masthead unit */}
      <Grid container className={classes.masthead} justify="center" alignItems="center">
        <Grid item md={8} sm={10} xs={12}>
          <Grid item md={6} sm={8} xs={12} >
            <Typography variant="h4" className={classes.mastheadText}>Invest smarter with Hugging Money.</Typography>
            <Typography variant="subtitle2" className={classes.mastheadText}>
              Join Hugging Money and get tools to help you build your own portfolio, without paying the management fee, redemption as like you buy mutual funds.
              </Typography>
            <br />
            <Button
              className={classes.mastheadButton}
              size="large"
              onClick={startOnClick}
            >
              Start Now
             </Button>
          </Grid>
        </Grid>
      </Grid>
      {/* End Masthead unit */}
      {/* Intro unit */}
      <Grid container className={classes.landingSectionLight} justify="center" alignItems="center">
        <Grid container item md={8} sm={10} xs={12} spacing={8}>
          <Grid item md={6} sm={6} xs={12} >
            <Typography variant="h5" >What is HuggingMoney?</Typography>
            <Typography variant="subtitle1" >
              Hugging Money helps people build their stock portfolio with several models. People can adjust the assets of the portfolio or change portfolio model in any time!
            </Typography>
          </Grid>
          <Grid item md={6} sm={6} xs={12} >
            <Typography variant="h5" >What is stock portfolio?</Typography>
            <Typography variant="subtitle1" >
              Stock portfolio is a collection of investments in stocks. A good portfolio can maximise the expected return and minimise the risk.
            </Typography>
          </Grid>
        </Grid>
      </Grid>
      {/* End Intro unit */}
      {/* Call-to-Action unit */}
      <Grid container className={classes.landingSectionDark} justify="center" alignItems="center" direction="column">
        <Grid item md={8} sm={10} xs={12} className={classes.sectionTitle}>
          <Typography variant="h4" >Why choose HuggingMoney?</Typography>
        </Grid>
        <Grid container item md={8} sm={10} xs={12} spacing={8}>
          <Grid item md={4} sm={4} xs={12} >
            <Grid container item direction="column" >
              <Box className={classes.iconContainer}>
                <CreateOutlinedIcon className={classes.icon} />
              </Box>
              <Typography className={classes.featureTitle} variant="h6" >Flexible</Typography>
              <Typography variant="subtitle1" >
                You can freely adjust your portfolio or change the portfolio model at any time, anywhere! Even when portfolio managers of other banks are on vacation!
            </Typography>
            </Grid>
          </Grid>
          <Grid item md={4} sm={4} xs={12} >
            <Grid container item direction="column" >
              <Box className={classes.iconContainer}>
                <ShowChartOutlinedIcon className={classes.icon} />
              </Box>
              <Typography className={classes.featureTitle} variant="h6" >Powerful</Typography>
              <Typography variant="subtitle1" >
                HuggingMoney provides you many powerful portfolio models, including our cutting edge model which can utilize the market news using AI techniques.
              </Typography>
            </Grid>
          </Grid>
          <Grid item md={4} sm={4} xs={12} >
            <Grid container item direction="column" >
              <Box className={classes.iconContainer}>
                <MoneyOffOutlinedIcon className={classes.icon} />
              </Box>
              <Typography className={classes.featureTitle} variant="h6" >No fees</Typography>
              <Typography variant="subtitle1" >
                Unlike investing in funds, you don't need to pay the ongoing charges, entry charges, redemption fees to us. You can hug more money!
              </Typography>
            </Grid>
          </Grid>
        </Grid>
      </Grid>
      {/* End Call-to-Action unit */}
      {/* Chart unit */}
      <Grid container className={classes.landingSectionLight} justify="center" alignItems="center">
        <Grid container item md={8} sm={10} xs={12} spacing={8}>
          <Grid item xl={4} md={12} xs={12} >
            <Typography variant="h5" >Excellent Performance</Typography>
            <Typography variant="subtitle1" >
              Our portfolios compare with Standard {'&'} Poor's 500(S{'&'}P 500).
            </Typography>
          </Grid>
          <Grid item xl={8} md={12} xs={12} >
            <Line data={setDemoDataset()}>
            </Line>
          </Grid>
        </Grid>
      </Grid>
      {/* End Chart unit */}
      {/* Price unit */}
      <div className={classes.landingSectionDark}>
        <Grid container justify="center" alignItems="center" direction="column">
          <Grid item md={8} sm={10} xs={12} className={classes.sectionTitle}>
            <Typography variant="h4" >Pricing</Typography>
          </Grid>
        </Grid>
        <Container maxWidth="md" component="main">
          <Grid container spacing={5} alignItems="flex-end">
            {tiers.map((tier) => (
              // Enterprise card is full width at sm breakpoint
              <Grid item key={tier.title} xs={12} sm={6} md={6}>
                <Card>
                  <CardHeader
                    title={tier.title}
                    subheader={tier.subheader}
                    titleTypographyProps={{ align: 'center' }}
                    subheaderTypographyProps={{ align: 'center' }}
                    action={tier.title === 'Pro' ? <StarIcon /> : null}
                    className={classes.cardHeader}
                  />
                  <CardContent>
                    <div className={classes.cardPricing}>
                      <Typography component="h2" variant="h3" color="textPrimary">
                        ${tier.price}
                      </Typography>
                      <Typography variant="h6" color="textSecondary">
                        /mo
                    </Typography>
                    </div>
                    <ul>
                      {tier.description.map((line) => (
                        <Typography component="li" variant="subtitle1" align="center" key={line}>
                          {line}
                        </Typography>
                      ))}
                    </ul>
                  </CardContent>
                  <CardActions>
                    <Button fullWidth variant={tier.buttonVariant} color="primary" onClick={()=>{window.location.pathname = './signup';}} >
                      {tier.buttonText}
                    </Button>
                  </CardActions>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Container>
      </div>
      {/* End Price unit */}
      {/* Footer */}
      <Container maxWidth="md" component="footer" className={classes.footer}>
        <Box>
          <Copyright />
        </Box>
      </Container>
      {/* End footer */}
    </div>
  )
}